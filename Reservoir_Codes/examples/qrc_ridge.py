"""QRC + ridge regression on a monitored random-Clifford reservoir (CAMPS).

Run from the MIPT_CODE folder:
    %run examples/qrc_ridge.py                 # default: 1 memory + 1 data qubit
    %run examples/qrc_ridge.py --quick         # ~20 s smoke test
    %run examples/qrc_ridge.py --n-mem 3       # bigger reservoir (brickwork U)
    python -m examples.qrc_ridge --help

Task: inputs theta_1..theta_K (theta = n pi/2^lam, n = 0..2^lam-1) are fed in
one per step; ridge regression on measurement-statistics features must
reconstruct all K inputs (theta_hat = W f + b).

Parts:
  0. checks  - CAMPS vs brute-force state vector; literal shots vs sampled records
  1. SWAP    - U = SWAP(mem, data): compare feature sets (note's lag-1, lag-2, all)
  2. random  - U_k independent random two-qubit Cliffords (frozen per reservoir),
               all singles + all two-time correlators, ridge picks; several
               reservoir realisations; also R^2 versus number of shots t
Outputs: examples/output/qrc/  (summary.txt, results.json, three figures)

Exact shortcut used for speed: for each input sequence the exact probability of
every measurement record is computed once by branching on each measurement
(2^(steps x data qubits) records). "t shots" are then t records drawn from that
distribution, which is statistically identical to running t trajectories;
part 0 checks this against literal trajectory-by-trajectory CAMPS runs.
"""
import argparse
import sys
import json
import time
from pathlib import Path

import numpy as np

_ROOT = str(__import__('pathlib').Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:          # allow `python examples/<script>.py` from anywhere
    sys.path.insert(0, _ROOT)
import matplotlib.pyplot as plt

from qrc import Reservoir, feature_names, features, shot_weights, fit_ridge, predict, r2_scores

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser(description="QRC + ridge regression with an online-CAMPS reservoir")
ap.add_argument("--n-mem", type=int, default=1, help="memory qubits (default 1)")
ap.add_argument("--n-data", type=int, default=1, help="data qubits (default 1)")
ap.add_argument("--inputs", type=int, default=3, help="inputs per sequence K (default 3)")
ap.add_argument("--flush", type=int, default=1, help="extra steps with no input (default 1)")
ap.add_argument("--lam", type=int, default=5, help="theta = n pi / 2^lam (default 5 -> 32 values)")
ap.add_argument("--cycles", type=int, default=1, help="brickwork cycles per U (default 1)")
ap.add_argument("--t-prob", type=float, default=0.0,
                help="probability of a T gate on each qubit after each Clifford (default 0: Clifford-only U)")
ap.add_argument("--n-values", type=int, default=32, help="number of input values n = 0..n_values-1 (default 32)")
ap.add_argument("--parallel", type=int, default=1,
                help="independent reservoirs fed the same inputs, features concatenated (default 1)")
ap.add_argument("--shots", type=int, default=200, help="shots t per input sequence (default 200)")
ap.add_argument("--train", type=int, default=600, help="training sequences (default 600)")
ap.add_argument("--test", type=int, default=300, help="test sequences (default 300)")
ap.add_argument("--realizations", type=int, default=3, help="random reservoirs to average (default 3)")
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--out", type=Path, default=HERE / "output" / "qrc")
ap.add_argument("--quick", action="store_true", help="small sizes for a fast check")
args = ap.parse_args()
if args.quick:
    args.train, args.test, args.realizations = 150, 80, 1
args.out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)
K = args.inputs
log_lines = []


def log(s=""):
    print(s, flush=True)
    log_lines.append(s)


def dataset(group, n_seq, rng, stats=None):
    """Random input sequences and, for every reservoir in the group, the exact
    record distribution of each sequence."""
    idx = rng.integers(0, group[0].n_values, size=(n_seq, group[0].n_inputs))
    thetas = group[0].angles[idx]
    dists = [[r.outcome_distribution(th, stats=stats) for r in group] for th in thetas]
    return thetas, dists


def feature_matrix(dists, pairs_list, t, rng):
    """Concatenated features of all reservoirs; t = None -> exact (infinite shots)."""
    rows = []
    for per_res in dists:
        rows.append(np.concatenate([features(M, w if t is None else shot_weights(w, t, rng), pairs)
                                    for (M, w), pairs in zip(per_res, pairs_list)]))
    return np.array(rows)


def evaluate(group, train, test, kind, t, rng):
    pairs = [feature_names(r, kind)[1] for r in group]
    Ftr = feature_matrix(train[1], pairs, t, rng)
    Fte = feature_matrix(test[1], pairs, t, rng)
    model = fit_ridge(Ftr, train[0], rng=rng)
    return r2_scores(test[0], predict(model, Fte)), model, Fte


def fmt(r2):
    return "  ".join(f"{v:6.3f}" for v in r2)


# ------------------------------------------------------------------ setup echo
n_total = args.n_mem + args.n_data
log("QRC + ridge regression, monitored random-Clifford reservoir (online CAMPS)")
log(f"qubits: {args.n_mem} memory + {args.n_data} data = {n_total};  K = {K} inputs + {args.flush} flush step(s)")
log(f"inputs theta = n*pi/2^{args.lam}, n = 0..{args.n_values - 1};  T-gate probability = {args.t_prob};  shots t = {args.shots};  "
    f"train/test sequences = {args.train}/{args.test}")
log("readout: theta_hat = W f + b, ridge regression, lambda by 5-fold CV;  score: test R^2 per theta_j")
results = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}
t0 = time.perf_counter()

# ------------------------------------------------------------------ part 0: checks
log("\n=== 0. Checks ===")
res0 = Reservoir(args.n_mem, args.n_data, K, args.flush, args.lam, "random", args.cycles, seed=args.seed,
                 t_prob=args.t_prob, n_values=args.n_values)
worst = 0.0
for th in res0.angles[rng.integers(0, args.n_values, size=(5, K))]:
    Mc, wc = res0.outcome_distribution(th, "camps")
    Md, wd = res0.outcome_distribution(th, "dense")
    dc = {tuple(m): p for m, p in zip(Mc, wc)}
    dd = {tuple(m): p for m, p in zip(Md, wd)}
    worst = max(worst, max(abs(dc.get(k, 0) - dd.get(k, 0)) for k in set(dc) | set(dd)))
log(f"CAMPS vs brute-force state vector, record probabilities (5 sequences): max |diff| = {worst:.1e}")
th = res0.angles[rng.integers(0, args.n_values, size=K)]
_, pairs_all = feature_names(res0, "all")
M, w = res0.outcome_distribution(th)
exact = features(M, w, pairs_all)
t_lit = 400 if args.quick else 2000
lit = res0.run_shots(th, t_lit, rng)
from_lit = features(lit, np.full(t_lit, 1 / t_lit), pairs_all)
z = np.abs(from_lit - exact) / np.sqrt(np.maximum(1 - exact ** 2, 1e-12) / t_lit)
log(f"{t_lit} literal CAMPS trajectories vs exact features: max deviation = {z.max():.2f} standard errors "
    f"({len(exact)} features; expect < ~3)")
results["checks"] = {"camps_vs_dense_max_diff": worst, "literal_shots_max_z": float(z.max())}

# ------------------------------------------------------------------ part 1: SWAP
log("\n=== 1. SWAP reservoir (1 memory + 1 data, U = SWAP every step) ===")
swap = [Reservoir(1, 1, K, args.flush, args.lam, "swap", n_values=args.n_values)]
train = dataset(swap, args.train, rng)
test = dataset(swap, args.test, rng)
kinds = ["lag1", "lag2", "all"]
labels = {"lag1": "singles + lag-1 (note)", "lag2": "singles + lag-2", "all": "singles + all pairs"}
log("test R^2 for " + "  ".join(f"theta_{j+1}" for j in range(K)))
swap_r2 = {}
for t in (args.shots, None):
    for kind in kinds:
        r2, model, _ = evaluate(swap, train, test, kind, t, rng)
        swap_r2[(kind, t)] = r2
        log(f"  t={'inf' if t is None else t:>4}  {labels[kind]:<24} {fmt(r2)}   (lambda={model['lambda']:.0e})")
results["swap"] = {f"{k}|t={'inf' if t is None else t}": v.tolist() for (k, t), v in swap_r2.items()}

# ------------------------------------------------------------------ part 2: random Clifford reservoirs
log(f"\n=== 2. Random-Clifford reservoir ({args.realizations} realisation(s) of {args.parallel} reservoir(s) "
    f"side by side, features: singles + all pairs) ===")
rand_r2 = {t: [] for t in (args.shots, None)}
camps_stats = []
first = None
for r in range(args.realizations):
    res = [Reservoir(args.n_mem, args.n_data, K, args.flush, args.lam, "random", args.cycles,
                     seed=args.seed + 1 + 100 * r + i, t_prob=args.t_prob, n_values=args.n_values)
           for i in range(args.parallel)]
    stats = {}
    train = dataset(res, args.train, rng, stats)
    test = dataset(res, args.test, rng, stats)
    camps_stats.append(stats)
    for t in rand_r2:
        r2, model, Fte = evaluate(res, train, test, "all", t, rng)
        rand_r2[t].append(r2)
        if r == 0 and t == args.shots:
            first = (res, train, test, model, Fte)
    log(f"  reservoir {r}: t={args.shots}: {fmt(rand_r2[args.shots][-1])}   t=inf: {fmt(rand_r2[None][-1])}   "
        f"OFD rot/meas {stats['rot_ofd']}/{stats['meas_ofd']}, MPS rot/meas {stats['rot_mps']}/{stats['meas_mps']}, "
        f"max bond {stats['max_bond']}")
for t, vals in rand_r2.items():
    a = np.array(vals)
    log(f"  mean over reservoirs, t={'inf' if t is None else t:>4}: "
        + "  ".join(f"{m:6.3f}+/-{s:.3f}" for m, s in zip(a.mean(0), a.std(0))))
results["random"] = {f"t={'inf' if t is None else t}": np.array(v).tolist() for t, v in rand_r2.items()}
results["camps_stats"] = camps_stats

# R^2 versus shots, reservoir 0
res, train, test, _, _ = first
shot_grid = [10, 30, 100, 300, 1000, 3000]
curve = {t: evaluate(res, train, test, "all", t, rng)[0] for t in shot_grid}
curve_inf = np.array(rand_r2[None][0])
log("\nR^2 versus shots t (reservoir 0): " + "; ".join(
    f"t={t}: {fmt(v)}" for t, v in curve.items()) + f"; t=inf: {fmt(curve_inf)}")
results["r2_vs_shots"] = {str(t): v.tolist() for t, v in curve.items()} | {"inf": curve_inf.tolist()}

# ------------------------------------------------------------------ figures
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.facecolor": BG, "figure.facecolor": BG,
                     "axes.spines.top": False, "axes.spines.right": False})

# (a) SWAP: R^2 per input for each feature set
fig, ax = plt.subplots(figsize=(8, 4.6))
width = 0.8 / len(kinds)
xs = np.arange(K)
for i, (c, kind) in enumerate(zip(COLORS, kinds)):
    vals = np.clip(swap_r2[(kind, args.shots)], 0, 1)
    ax.bar(xs + (i - (len(kinds) - 1) / 2) * width, vals, width * 0.92, color=c, label=labels[kind])
ax.set_xticks(xs); ax.set_xticklabels([f"θ{j+1}" for j in range(K)])
ax.set_ylim(0, 1.05); ax.grid(axis="y", color=GRID, lw=0.8)
ax.set(ylabel="Test R²", title=f"SWAP reservoir: which correlators carry the inputs (t = {args.shots} shots)")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
fig.tight_layout(); fig.savefig(args.out / "1_swap_feature_sets.png", dpi=170); plt.show()

# (b) random reservoir 0: predicted vs true
res, train, test, model, Fte = first
pred = predict(model, Fte)
fig, axs = plt.subplots(1, K, figsize=(4.2 * K, 4.2), squeeze=False)
for j, ax in enumerate(axs[0]):
    top = float(res[0].angles.max())
    ax.plot([0, top], [0, top], ls="--", color=INK2, lw=1)
    ax.scatter(test[0][:, j], pred[:, j], s=12, color=COLORS[0], alpha=0.6, edgecolors="none")
    ax.set(xlabel=f"true θ{j+1}", ylabel=f"predicted θ{j+1}" if j == 0 else "",
           title=f"θ{j+1}: R² = {rand_r2[args.shots][0][j]:.3f}")
    ax.set_xlim(-0.1 * top, 1.1 * top); ax.set_ylim(-0.3 * top, 1.3 * top); ax.grid(color=GRID, lw=0.8)
fig.suptitle(f"Random-Clifford reservoir 0 ({args.parallel} x {n_total} qubits), test set, t = {args.shots} shots", y=1.0)
fig.tight_layout(); fig.savefig(args.out / "2_random_pred_vs_true.png", dpi=170); plt.show()

# (c) R^2 vs shots
fig, ax = plt.subplots(figsize=(8, 4.6))
for j, c in zip(range(K), COLORS):
    ys = [curve[t][j] for t in shot_grid]
    ax.plot(shot_grid, ys, color=c, lw=2, marker="o", ms=6, mec=BG, mew=1.2, label=f"θ{j+1}")
    ax.axhline(curve_inf[j], color=c, lw=1, ls=":")
ax.set_xscale("log"); ax.set_ylim(min(0, min(min(v) for v in curve.values())) - 0.02, 1.02)
ax.grid(color=GRID, lw=0.8, which="both")
ax.set(xlabel="Shots t per input sequence", ylabel="Test R²",
       title="Shot noise: R² versus shots (dotted = infinitely many shots)")
ax.legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(args.out / "3_r2_vs_shots.png", dpi=170); plt.show()

log(f"\nfinished in {time.perf_counter() - t0:.0f} s; outputs in {args.out}")
(args.out / "summary.txt").write_text("\n".join(log_lines) + "\n")
(args.out / "results.json").write_text(json.dumps(results, indent=1, default=float))
