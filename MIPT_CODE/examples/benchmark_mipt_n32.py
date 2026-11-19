#!/usr/bin/env python3
"""Benchmark the existing Clifford-MIPT trajectory simulator at N=32.

Default benchmark:
  * N = 32 qubits
  * depth = 4*N full brickwork cycles (same convention as hpc/config.json)
  * 10 independent trajectories at each p = 0.05, 0.50, 0.95

The first measured trajectory at each p records the half-chain entropy history,
matching the extra diagnostic work performed for the first trajectory of each
condition in the production hpc_batch.py workflow. Clifford-group enumeration
and a separate warm-up trajectory are timed separately and excluded from the
per-trajectory timing statistics.

Outputs (CSV, JSON, PNG) are written to examples/output/mipt_benchmarks/.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from mipt.clifford_mipt import run_trajectory, two_qubit_clifford_words


DEFAULT_PROBABILITIES = [0.05, 0.50, 0.95]
PRODUCTION_PROBABILITIES = [
    0.0, 0.05, 0.10, 0.15, 0.20,
    0.25, 0.30, 0.40, 0.50, 0.70, 1.0,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Time N=32 monitored Clifford trajectories and estimate local runtime."
    )
    parser.add_argument("--n", type=int, default=32, help="Number of qubits (default: 32).")
    parser.add_argument(
        "--depth-factor", type=int, default=4,
        help="Full brickwork cycles per qubit; depth = factor*N (default: 4).",
    )
    parser.add_argument(
        "--trajectories", type=int, default=10,
        help="Measured trajectories at EACH benchmark probability (default: 10).",
    )
    parser.add_argument(
        "--probabilities", type=float, nargs="+", default=DEFAULT_PROBABILITIES,
        help="Measurement probabilities to benchmark (default: 0.05 0.50 0.95).",
    )
    parser.add_argument("--seed", type=int, default=20261117, help="Base random seed.")
    parser.add_argument(
        "--target-trajectories-per-condition", type=int, default=20,
        help="Planned trajectories per (N,p) on each computer (default: 20).",
    )
    parser.add_argument(
        "--production-probabilities", type=float, nargs="+",
        default=PRODUCTION_PROBABILITIES,
        help="p grid used by the five-computer production experiment.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("examples/output/mipt_benchmarks"),
        help="Directory for benchmark CSV/JSON/PNG outputs.",
    )
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    if args.n < 2 or args.n % 2:
        raise ValueError("n must be an even integer >= 2 for the half-chain entropy benchmark.")
    if args.depth_factor < 1:
        raise ValueError("depth-factor must be >= 1.")
    if args.trajectories < 2:
        raise ValueError("Use at least 2 measured trajectories per probability.")
    if args.target_trajectories_per_condition < 1:
        raise ValueError("target-trajectories-per-condition must be >= 1.")
    for p in list(args.probabilities) + list(args.production_probabilities):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Measurement probability must lie in [0,1], got {p}.")
    if len(set(args.probabilities)) != len(args.probabilities):
        raise ValueError("Benchmark probabilities must be unique.")


def save_plot(rows: list[dict], output_path: Path, n: int, depth: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ps = sorted({float(row["p"]) for row in rows})
    means, sems = [], []
    for p in ps:
        vals = [float(row["seconds"]) for row in rows if float(row["p"]) == p]
        means.append(statistics.mean(vals))
        sems.append(statistics.stdev(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    ax.errorbar(ps, means, yerr=sems, fmt="o-", capsize=4, linewidth=1.8)
    ax.set_xlabel("Measurement probability p")
    ax.set_ylabel("Wall-clock seconds per trajectory")
    ax.set_title(f"N={n} Clifford-MIPT trajectory benchmark (depth={depth} cycles)")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    validate(args)
    n = args.n
    depth = args.depth_factor * n
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"benchmark_N{n}_{timestamp}"
    raw_csv = output_dir / f"{stem}_trajectories.csv"
    summary_csv = output_dir / f"{stem}_summary.csv"
    metadata_json = output_dir / f"{stem}_metadata.json"
    plot_png = output_dir / f"{stem}_runtime.png"

    print("=" * 72, flush=True)
    print("CLIFFORD-MIPT RUNTIME BENCHMARK", flush=True)
    print(f"System: {platform.platform()}", flush=True)
    print(f"Python: {sys.version.split()[0]} | NumPy: {np.__version__}", flush=True)
    print(f"Qubits: N={n}", flush=True)
    print(f"Depth: {depth} full brickwork cycles = {2*depth} two-qubit gate layers", flush=True)
    print(f"Measured trajectories: {args.trajectories} at each p={args.probabilities}", flush=True)
    print("=" * 72, flush=True)

    # The one-time BFS enumeration of all 11,520 two-qubit Cliffords is not
    # repeated for each trajectory. Measure it separately, then exclude it
    # from the steady-state per-trajectory timings.
    t0 = time.perf_counter()
    words = two_qubit_clifford_words()
    clifford_setup_seconds = time.perf_counter() - t0
    print(f"Clifford library: {len(words)} gates; setup {clifford_setup_seconds:.3f} s", flush=True)

    # Warm up NumPy and the trajectory code; don't include this in the sample.
    warmup_rng = np.random.default_rng(np.random.SeedSequence([args.seed, n, 987654321]))
    t0 = time.perf_counter()
    run_trajectory(n, depth, float(args.probabilities[len(args.probabilities)//2]),
                   warmup_rng, words, record=False)
    warmup_seconds = time.perf_counter() - t0
    print(f"Warm-up trajectory (excluded): {warmup_seconds:.3f} s", flush=True)

    rows: list[dict] = []
    with raw_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["N", "depth_cycles", "p", "trajectory", "seconds",
                        "entropy_bits", "measurements", "random_outcomes", "recorded_history"],
        )
        writer.writeheader()
        for p_index, p in enumerate(args.probabilities):
            print(f"\nBenchmarking N={n}, p={p:.4f}", flush=True)
            for trajectory in range(args.trajectories):
                # Match production: the first trajectory for each condition
                # records entropy at every half-layer; other trajectories do not.
                record_history = trajectory == 0
                seed = np.random.SeedSequence(
                    [args.seed, n, int(round(p * 1_000_000)), trajectory, p_index]
                )
                rng = np.random.default_rng(seed)
                start = time.perf_counter()
                result = run_trajectory(n, depth, float(p), rng, words, record=record_history)
                elapsed = time.perf_counter() - start
                row = {
                    "N": n,
                    "depth_cycles": depth,
                    "p": f"{p:.6f}",
                    "trajectory": trajectory,
                    "seconds": f"{elapsed:.6f}",
                    "entropy_bits": result["entropy"],
                    "measurements": result["measurements"],
                    "random_outcomes": result["random_outcomes"],
                    "recorded_history": int(record_history),
                }
                writer.writerow(row)
                handle.flush()
                rows.append({**row, "p": float(p), "seconds": elapsed})
                print(
                    f"  trajectory {trajectory + 1:>2}/{args.trajectories}: "
                    f"{elapsed:.3f} s; S_A={result['entropy']} bits; "
                    f"measurements={result['measurements']}",
                    flush=True,
                )

    summary_rows = []
    mean_by_p: dict[float, float] = {}
    for p in sorted({float(row["p"]) for row in rows}):
        subset = [float(row["seconds"]) for row in rows if float(row["p"]) == p]
        mean_s = statistics.mean(subset)
        median_s = statistics.median(subset)
        sd_s = statistics.stdev(subset) if len(subset) > 1 else 0.0
        sem_s = sd_s / np.sqrt(len(subset))
        mean_by_p[p] = mean_s
        summary_rows.append({
            "N": n, "depth_cycles": depth, "p": p, "trajectories": len(subset),
            "mean_seconds": mean_s, "median_seconds": median_s,
            "sd_seconds": sd_s, "sem_seconds": sem_s,
            "min_seconds": min(subset), "max_seconds": max(subset),
        })

    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Project the planned local shard: 20 trajectories at each of the 11
    # production p values for this N. Use the nearest measured p as a proxy.
    # This is an estimate, not a guarantee; benchmark all production p values
    # later if p-dependent timing differences are substantial.
    projected_rows = []
    for target_p in args.production_probabilities:
        nearest_p = min(mean_by_p, key=lambda measured_p: abs(measured_p - target_p))
        seconds_per_trajectory = mean_by_p[nearest_p]
        projected = args.target_trajectories_per_condition * seconds_per_trajectory
        projected_rows.append({
            "production_p": float(target_p),
            "timing_proxy_p": nearest_p,
            "trajectories": args.target_trajectories_per_condition,
            "estimated_seconds": projected,
        })
    projected_total = sum(row["estimated_seconds"] for row in projected_rows)

    metadata = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "N": n,
        "depth_factor": args.depth_factor,
        "depth_cycles": depth,
        "two_qubit_gate_layers": 2 * depth,
        "benchmark_probabilities": args.probabilities,
        "trajectories_per_benchmark_probability": args.trajectories,
        "production_probabilities": args.production_probabilities,
        "target_trajectories_per_production_probability": args.target_trajectories_per_condition,
        "projected_trajectories_for_N": len(args.production_probabilities) * args.target_trajectories_per_condition,
        "clifford_library_size": len(words),
        "clifford_library_setup_seconds_excluded": clifford_setup_seconds,
        "warmup_seconds_excluded": warmup_seconds,
        "timing_method": "time.perf_counter around run_trajectory; first trajectory at each p records entropy history as in production",
        "projection_method": "For each production p, use mean measured time at nearest benchmark p, multiplied by planned trajectories per condition; sum across p grid.",
        "projected_compute_seconds_for_N": projected_total,
        "projected_compute_hours_for_N": projected_total / 3600.0,
        "note": "Projection is approximate. It excludes Jupyter/OS contention, and assumes benchmark representative p timing and a single trajectory stream. Setup and warm-up excluded.",
    }
    with metadata_json.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    save_plot(rows, plot_png, n, depth)

    print("\n" + "=" * 72, flush=True)
    print("BENCHMARK SUMMARY", flush=True)
    for row in summary_rows:
        print(
            f"p={row['p']:.3f}: mean={row['mean_seconds']:.3f} s, "
            f"median={row['median_seconds']:.3f} s, "
            f"SD={row['sd_seconds']:.3f} s",
            flush=True,
        )
    print("\nEstimated N=32 shard workload:", flush=True)
    print(f"  {args.target_trajectories_per_condition} trajectories at each of "
          f"{len(args.production_probabilities)} production probabilities", flush=True)
    print(f"  Total trajectories: {len(projected_rows) * args.target_trajectories_per_condition}", flush=True)
    print(f"  Estimated compute time: {projected_total / 3600.0:.2f} hours "
          f"({projected_total / 60.0:.1f} minutes)", flush=True)
    print("\nFiles written:", flush=True)
    for path in (raw_csv, summary_csv, metadata_json, plot_png):
        print(f"  {path}", flush=True)
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
