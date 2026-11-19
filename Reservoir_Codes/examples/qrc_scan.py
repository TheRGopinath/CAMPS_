"""Scan reservoir size: QRC recall task for several (memory, data) qubit counts.

Same circuit and readout as examples/qrc_ridge.py, but for bigger registers,
using literal shots (t full CAMPS trajectories per input sequence) on all CPU
cores. Every step: RY(theta_k) on every data qubit, U_k = one brickwork cycle
of independent random two-qubit Cliffords (frozen per reservoir), Z measurement
of every data qubit. Measurement rate = data qubits / total qubits.

Run from the MIPT_CODE folder (in Jupyter, Linux or Windows):
    %run examples/qrc_scan.py --quick          # ~1-2 min smoke test
    %run examples/qrc_scan.py --laptop         # laptop preset (~40-60 min on 4 cores), see below
    %run examples/qrc_scan.py                  # full default grid (~1-1.5 h on 16 cores)
    %run examples/qrc_scan.py --configs 4x4 6x2 --realizations 5
    %run examples/qrc_scan.py --workers 8      # limit cores
Interrupted? Run the same command again: finished (config, reservoir) groups
are kept and skipped. Re-running a finished scan just rebuilds the summary.

Config "AxB" = A memory + B data qubits. Default grid:
    1x1 2x2 3x3 4x4   (rate 1/2, growing)
    3x1 6x2           (rate 1/4)
    1x3 2x6           (rate 3/4)

Laptop preset (--laptop): the five configs that answer both questions cheaply,
    size at rate 1/2:          1x1 -> 2x2 -> 3x3
    rate at 4 qubits:          3x1 (1/4) -> 2x2 (1/2) -> 1x3 (3/4)
    3 reservoirs each, 500 train + 200 test sequences, 100 shots, 4 workers,
    output in examples/output/qrc_scan_laptop/.

Outputs (default examples/output/qrc_scan/):
    features/<config>_r<k>.npz   thetas, features, CAMPS stats per reservoir
    scan_results.csv             one row per (config, reservoir): test R^2 per input
    summary.txt, RUN_NOTES.txt, scan_r2.png
Send me the whole folder (or at least scan_results.csv + summary.txt).
"""
import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = str(__import__('pathlib').Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:          # allow `python examples/<script>.py` from anywhere
    sys.path.insert(0, _ROOT)

from qrc import feature_names, fit_ridge, predict, r2_scores
from qrc.scan_worker import job, make_reservoir as _make_reservoir, parse

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "output" / "qrc_scan"
DEFAULT_CONFIGS = ["1x1", "2x2", "3x3", "4x4", "3x1", "6x2", "1x3", "2x6"]

ap = argparse.ArgumentParser(description="QRC recall task versus reservoir size (online CAMPS, literal shots)")
ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS, help="memory x data, e.g. 4x4 6x2")
ap.add_argument("--layout", choices=["interleaved", "block"], default="interleaved",
                help="interleaved: data qubits spread along the chain (default); block: memory then data")
ap.add_argument("--realizations", type=int, default=3, help="random reservoirs per config (default 3)")
ap.add_argument("--inputs", type=int, default=3, help="inputs per sequence K (default 3)")
ap.add_argument("--flush", type=int, default=1, help="extra steps with no input (default 1)")
ap.add_argument("--lam", type=int, default=5)
ap.add_argument("--n-values", type=int, default=32)
ap.add_argument("--t-prob", type=float, default=0.0, help="T-gate probability inside U (default 0)")
ap.add_argument("--shots", type=int, default=200, help="shots t per input sequence (default 200)")
ap.add_argument("--train", type=int, default=800, help="training sequences (default 800)")
ap.add_argument("--test", type=int, default=300, help="test sequences (default 300)")
ap.add_argument("--seed", type=int, default=20261118)
ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
ap.add_argument("--quick", action="store_true", help="tiny run: configs 1x1 2x2, 1 reservoir, 120/60 sequences, 100 shots")
ap.add_argument("--laptop", action="store_true",
                help="laptop preset: configs 1x1 2x2 3x3 3x1 1x3, 3 reservoirs, 500/200 sequences, 100 shots, 4 workers")
ap.add_argument("--start-method", choices=["auto", "fork", "spawn"], default="auto",
                help="multiprocessing start method (auto: fork on Linux/macOS, spawn on Windows)")
args = ap.parse_args()
def _preset(**values):
    """Apply preset values, except for options the user typed explicitly."""
    for key, value in values.items():
        if "--" + key.replace("_", "-") not in sys.argv:
            setattr(args, key, value)


if args.quick:
    _preset(configs=["1x1", "2x2"], realizations=1, train=120, test=60, shots=100)
    if args.out == DEFAULT_OUT:
        args.out = HERE / "output" / "qrc_scan_quick"
elif args.laptop:
    _preset(configs=["1x1", "2x2", "3x3", "3x1", "1x3"], realizations=3, train=500, test=200, shots=100,
            workers=min(4, os.cpu_count() or 1))
    if args.out == DEFAULT_OUT:
        args.out = HERE / "output" / "qrc_scan_laptop"
(args.out / "features").mkdir(parents=True, exist_ok=True)
K = args.inputs


PARAMS = {"inputs": K, "flush": args.flush, "lam": args.lam, "n_values": args.n_values,
          "t_prob": args.t_prob, "layout": args.layout, "seed": args.seed, "shots": args.shots}


def make_reservoir(cfg, r):
    return _make_reservoir(cfg, r, PARAMS)


def run_group(pool, cfg, r):
    path = args.out / "features" / f"{cfg}_r{r}.npz"
    if path.exists():
        return path, False
    res = make_reservoir(cfg, r)
    nm, nd = parse(cfg)
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, nm, nd, r, 1]))
    n_seq = args.train + args.test
    thetas = res.angles[rng.integers(0, res.n_values, size=(n_seq, K))]
    names = feature_names(res, "all")[0]
    F = np.zeros((n_seq, len(names)))
    totals = {}
    t0 = time.perf_counter(); last = t0
    tasks = [(cfg, r, i, thetas[i], PARAMS) for i in range(n_seq)]
    for done, (i, f, st) in enumerate(pool.imap_unordered(job, tasks, chunksize=4), 1):
        F[i] = f
        for k, v in st.items():
            totals[k] = max(totals.get(k, 0), v) if k == "max_bond" else totals.get(k, 0) + v
        now = time.perf_counter()
        if now - last > 20 or done == n_seq:
            eta = (now - t0) / done * (n_seq - done)
            print(f"    {cfg} reservoir {r}: {done}/{n_seq} sequences ({(now-t0)/60:.1f} min, ~{eta/60:.1f} min left)",
                  flush=True)
            last = now
    np.savez(path, thetas=thetas, features=F, names=np.array(names),
             data_qubits=np.array(res.data), stats_keys=np.array(list(totals)),
             stats_vals=np.array(list(totals.values())), seconds=time.perf_counter() - t0)
    return path, True


def analyse(plot=False):
    rows = []
    for cfg in args.configs:
        nm, nd = parse(cfg)
        for r in range(args.realizations):
            path = args.out / "features" / f"{cfg}_r{r}.npz"
            if not path.exists():
                continue
            d = np.load(path)
            th, F = d["thetas"], d["features"]
            ntr = args.train
            model = fit_ridge(F[:ntr], th[:ntr], rng=np.random.default_rng(0))
            r2 = r2_scores(th[ntr:], predict(model, F[ntr:]))
            stats = dict(zip(d["stats_keys"].tolist(), d["stats_vals"].tolist()))
            rows.append({"config": cfg, "n_mem": nm, "n_data": nd, "n_qubits": nm + nd,
                         "measurement_rate": round(nd / (nm + nd), 3), "layout": args.layout,
                         "reservoir": r, **{f"R2_theta{j+1}": round(float(v), 4) for j, v in enumerate(r2)},
                         "mean_R2": round(float(np.mean(r2)), 4), "recovered_R2_gt_0.5": int((r2 > 0.5).sum()),
                         "n_features": F.shape[1], "lambda": model["lambda"], "max_bond": stats.get("max_bond", 1),
                         "rot_ofd": stats.get("rot_ofd", 0), "rot_mps": stats.get("rot_mps", 0),
                         "meas_ofd": stats.get("meas_ofd", 0), "meas_mps": stats.get("meas_mps", 0),
                         "seconds": round(float(d["seconds"]), 1)})
    if not rows:
        return
    with (args.out / "scan_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines = [f"QRC recall scan: K = {K} inputs, {args.flush} flush, t = {args.shots} shots, "
             f"train/test = {args.train}/{args.test}, layout = {args.layout}, T-prob = {args.t_prob}",
             "config  qubits  rate   mean R^2 per input over reservoirs           recovered   max bond"]
    summary = []
    for cfg in args.configs:
        sel = [x for x in rows if x["config"] == cfg]
        if not sel:
            continue
        r2 = np.array([[x[f"R2_theta{j+1}"] for j in range(K)] for x in sel])
        rec = sum(x["recovered_R2_gt_0.5"] for x in sel)
        lines.append(f"{cfg:<7} {sel[0]['n_qubits']:>4}   {sel[0]['measurement_rate']:.2f}   "
                     + "  ".join(f"{m:5.2f}" for m in r2.mean(0)) + f"   (all: {r2.mean():.2f})"
                     + f"      {rec}/{r2.size}       {max(x['max_bond'] for x in sel)}")
        summary.append((cfg, sel[0]["measurement_rate"], r2))
    text = "\n".join(lines)
    (args.out / "summary.txt").write_text(text + "\n")
    if not plot:
        return
    print("\n" + text)

    import matplotlib.pyplot as plt
    INK2, GRID, BG = "#52514e", "#e4e3df", "#fcfcfb"
    COLORS = {0.25: "#eb6834", 0.5: "#2a78d6", 0.75: "#1baf7a"}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.facecolor": BG, "figure.facecolor": BG,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for x, (cfg, rate, r2) in enumerate(summary):
        c = COLORS.get(round(rate, 2), "#6250d6")
        ax.bar(x, np.clip(r2.mean(), 0, 1), 0.65, color=c, alpha=0.85)
        ax.scatter(np.full(r2.size, x) + np.linspace(-0.2, 0.2, r2.size), np.clip(r2.ravel(), -0.05, 1),
                   s=14, color="#0b0b0b", zorder=3)
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels([f"{c}\n({int(parse(c)[0])+int(parse(c)[1])} q)" for c, _, _ in summary])
    for rate, c in COLORS.items():
        ax.bar([], [], color=c, label=f"measurement rate {rate:g}")
    ax.set_ylim(-0.08, 1.05); ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set(xlabel="memory x data qubits", ylabel="test R²  (bar = mean, dots = each input & reservoir)",
           title=f"Recall of {K} inputs vs reservoir size (t = {args.shots} shots)")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(args.out / "scan_r2.png", dpi=170); plt.show()


if __name__ == "__main__":
    with (args.out / "RUN_NOTES.txt").open("w") as f:
        f.write("QRC reservoir-size scan (examples/qrc_scan.py)\ncommand: " + " ".join(sys.argv) + "\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
    print(f"configs {args.configs}, {args.realizations} reservoir(s) each, layout {args.layout}, "
          f"{args.train}+{args.test} sequences x {args.shots} shots, {args.workers} worker(s) -> {args.out}")
    method = args.start_method
    if method == "auto":
        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    ctx = mp.get_context(method)
    started = time.perf_counter()
    with ctx.Pool(args.workers) as pool:
        for cfg in args.configs:
            for r in range(args.realizations):
                path, ran = run_group(pool, cfg, r)
                print(f"  {cfg} reservoir {r}: {'done' if ran else 'already on disk'} -> {path.name}", flush=True)
                analyse()      # refresh CSV + summary after every finished group
    analyse(plot=True)
    print(f"\nfinished in {(time.perf_counter() - started) / 60:.1f} min; results in {args.out}")
