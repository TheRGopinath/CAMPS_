"""Fine-grid MIPT scan for finite-size scaling (Li-Chen-Fisher style).

Same physics as examples/mipt_clifford.py (same run_trajectory): brickwork of
uniform random 2-qubit Cliffords, Z measurement attempt with probability p on
every site after every gate layer, Born-rule outcomes, half-chain entropy at
the end of depth = depth_factor*N full cycles, starting from |0...0>.

What is new here:
  * a fine p grid around p_c ~ 0.16, including p = 0.16 itself;
  * larger sizes and more trajectories by default;
  * runs on all CPU cores (multiprocessing);
  * every trajectory is written to disk as soon as it finishes, so an
    interrupted run can be RESUMED by running the same command again;
  * each trajectory has its own reproducible seed derived from
    (seed, N, p, trajectory index), so results do not depend on core count
    or run order.

In Jupyter, with the notebook in the repository root (the folder containing
mipt/ and examples/):

    %run examples/mipt_scan.py --quick                # 1-minute smoke test
    %run examples/mipt_scan.py                        # full default scan
    %run examples/mipt_scan.py --workers 4            # limit cores

From a terminal in the repository root:

    python -m examples.mipt_scan                      # full default scan
    python -m examples.mipt_scan --quick              # 1-minute smoke test
    python -m examples.mipt_scan --workers 8          # choose core count
    python -m examples.mipt_scan --depth-factor 8 --sizes 32 --probs 0.10 0.16 0.25 --out examples/output/depth_check

Outputs in --out (default: output/mipt_scan/ next to this file):
  raw_trajectories.csv   one row per trajectory (appended live)
  entanglement_vs_p.csv  per-(N,p) mean, standard error, ... (written at the end)
  RUN_NOTES.txt          the exact settings used
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np

from mipt.clifford_mipt import run_trajectory, two_qubit_clifford_words

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / 'output' / 'mipt_scan'
DEFAULT_SIZES = [16, 24, 32, 48, 64]
DEFAULT_PROBS = [0.00, 0.05, 0.08, 0.10, 0.12, 0.13, 0.14, 0.15, 0.16,
                 0.17, 0.18, 0.19, 0.20, 0.22, 0.25, 0.30, 0.40, 0.50]
RAW_FIELDS = ['N', 'depth_cycles', 'p', 'trajectory', 'seed', 'entropy_bits',
              'measurement_count', 'random_outcomes', 'runtime_seconds']
SUMMARY_FIELDS = ['N', 'depth_cycles', 'p', 'trajectories', 'mean_entropy_bits',
                  'sem_bits', 'mean_entropy_density', 'mean_measurement_count']


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--sizes', type=int, nargs='+', default=DEFAULT_SIZES)
    ap.add_argument('--probs', type=float, nargs='+', default=DEFAULT_PROBS)
    ap.add_argument('--trajectories', type=int, default=200,
                    help='trajectories per (N, p) (default 200)')
    ap.add_argument('--depth-factor', type=int, default=4,
                    help='full cycles = depth_factor * N (default 4)')
    ap.add_argument('--seed', type=int, default=20261117)
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 1,
                    help='parallel processes (default: all cores)')
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--quick', action='store_true',
                    help='tiny smoke test: sizes 8 16, 5 probabilities, 10 trajectories')
    args = ap.parse_args()
    if args.quick:
        args.sizes, args.probs, args.trajectories = [8, 16], [0.0, 0.1, 0.16, 0.2, 0.5], 10
        if args.out == DEFAULT_OUT:
            args.out = HERE / 'output' / 'mipt_scan_quick'
    return args


def p_key(p):
    """Integer label for p (in units of 1e-4) so floats compare exactly."""
    return int(round(p * 10000))


def task_seed(base, n, p, traj):
    """Independent, reproducible 32-bit seed for one trajectory."""
    return int(np.random.SeedSequence([base, n, p_key(p), traj]).generate_state(1)[0])


_WORDS = None


def _init_worker():
    global _WORDS
    _WORDS = two_qubit_clifford_words()   # built once per process (~0.1 s)


def _run_one(task):
    n, depth, p, traj, seed = task
    t0 = time.perf_counter()
    res = run_trajectory(n, depth, p, np.random.default_rng(seed), _WORDS)
    return {'N': n, 'depth_cycles': depth, 'p': p, 'trajectory': traj, 'seed': seed,
            'entropy_bits': res['entropy'], 'measurement_count': res['measurements'],
            'random_outcomes': int(res['random_outcomes']),
            'runtime_seconds': round(time.perf_counter() - t0, 5)}


def load_done(raw_path, depth_factor):
    """(N, p_key, trajectory) triples already on disk, for resuming."""
    done = set()
    if raw_path.exists():
        with raw_path.open() as f:
            for r in csv.DictReader(f):
                n = int(r['N'])
                if int(r['depth_cycles']) == depth_factor * n:
                    done.add((n, p_key(float(r['p'])), int(r['trajectory'])))
    return done


def summarise(raw_path, out_path, sizes, probs, depth_factor):
    groups = {}
    with raw_path.open() as f:
        for r in csv.DictReader(f):
            n, pk = int(r['N']), p_key(float(r['p']))
            if n in sizes and pk in {p_key(p) for p in probs} and int(r['depth_cycles']) == depth_factor * n:
                groups.setdefault((n, pk), []).append(
                    (float(r['entropy_bits']), float(r['measurement_count'])))
    rows = []
    for (n, pk) in sorted(groups):
        ent = np.array([g[0] for g in groups[(n, pk)]]); meas = [g[1] for g in groups[(n, pk)]]
        sem = float(ent.std(ddof=1) / np.sqrt(len(ent))) if len(ent) > 1 else 0.0
        rows.append({'N': n, 'depth_cycles': depth_factor * n, 'p': pk / 10000,
                     'trajectories': len(ent), 'mean_entropy_bits': float(ent.mean()),
                     'sem_bits': sem, 'mean_entropy_density': float(ent.mean()) / (n / 2),
                     'mean_measurement_count': float(np.mean(meas))})
    with out_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS); w.writeheader(); w.writerows(rows)
    return rows


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raw_path = args.out / 'raw_trajectories.csv'
    done = load_done(raw_path, args.depth_factor)

    tasks = []
    for n in sorted(args.sizes, reverse=True):          # big N first: better load balance
        depth = args.depth_factor * n
        for p in args.probs:
            for traj in range(args.trajectories):
                if (n, p_key(p), traj) not in done:
                    tasks.append((n, depth, float(p), traj, task_seed(args.seed, n, p, traj)))

    with (args.out / 'RUN_NOTES.txt').open('w') as f:
        f.write('Clifford MIPT fine-grid scan (examples/mipt_scan.py)\n')
        f.write(f'command: {" ".join(sys.argv)}\n')
        for k in ('sizes', 'probs', 'trajectories', 'depth_factor', 'seed', 'workers'):
            f.write(f'{k}: {getattr(args, k)}\n')
        f.write('Protocol: |0..0>, open chain; each full cycle = odd-bond uniform 2-qubit Cliffords, '
                'Z-measurement attempt on every site with prob p, even-bond Cliffords, '
                'measurement attempt again. Born-rule outcomes. Half-chain entropy '
                '(qubits 0..N/2-1) at the end, from GF(2) rank, in bits.\n')

    total = len(tasks)
    print(f'{len(done)} trajectories already on disk; {total} to run on {args.workers} worker(s).')
    print(f'Sizes {args.sizes}, {len(args.probs)} probabilities, {args.trajectories} trajectories each, '
          f'depth = {args.depth_factor}N cycles. Output: {args.out}/')
    if total:
        new_file = not raw_path.exists()
        start = time.perf_counter(); last = 0.0
        # 'fork' (Linux/macOS) lets the workers see this script's functions even
        # under Jupyter's %run; Windows only has 'spawn', so use the terminal there.
        ctx = mp.get_context('fork' if 'fork' in mp.get_all_start_methods() else 'spawn')
        with raw_path.open('a', newline='') as f, ctx.Pool(args.workers, initializer=_init_worker) as pool:
            writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
            if new_file:
                writer.writeheader()
            for i, row in enumerate(pool.imap_unordered(_run_one, tasks, chunksize=4), 1):
                writer.writerow(row)
                now = time.perf_counter() - start
                if now - last > 15 or i == total:
                    f.flush(); last = now
                    eta = now / i * (total - i)
                    print(f'  {i}/{total} done  ({now/60:.1f} min elapsed, ~{eta/60:.1f} min left)', flush=True)

    rows = summarise(raw_path, args.out / 'entanglement_vs_p.csv', args.sizes, args.probs, args.depth_factor)
    print('\nN    p     <S_half> +/- sem   (bits)')
    for r in rows:
        print(f"{r['N']:<4} {r['p']:.2f}  {r['mean_entropy_bits']:7.3f} +/- {r['sem_bits']:.3f}   "
              f"(n={r['trajectories']})")
    print(f'\nSaved {args.out}/entanglement_vs_p.csv and raw_trajectories.csv')


if __name__ == '__main__':
    main()
