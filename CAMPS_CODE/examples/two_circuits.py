"""Run two seeded Clifford+T circuits: OFD succeeds and OFD encounters failures.

Run from the repository root: python -m examples.two_circuits
"""
from time import perf_counter
import numpy as np
from stabilizers_pbc import Gate, Pauli
from camps import simulate, move_t_gates
from camps.reference import exact_state, exact_expectation
from .draw_circuit import draw_circuit


def candidate(rng, n, t_count):
    """Randomly interleave H/S/CX and T, with entangling Clifford layers."""
    circuit = []
    for _ in range(t_count):
        for _ in range(2):
            kind = rng.choice(["H", "S", "CX"])
            a = int(rng.integers(n))
            b = (a + int(rng.integers(1, n))) % n
            circuit.append(Gate(kind, a, b) if kind == "CX" else Gate(kind, a))
        circuit.append(Gate("T", int(rng.integers(n))))
    return circuit


def planted_success(rng, n, t_count):
    """Random Clifford+T layout with a fresh H/T pivot at each step."""
    sites = rng.permutation(n)[:t_count]
    circuit = []
    for site in sites:
        site = int(site)
        circuit.append(Gate("H", site))
        if rng.integers(2):
            circuit.append(Gate("S", site))
        circuit.append(Gate("T", site))
        other = (site + int(rng.integers(1, n))) % n
        circuit.append(Gate("CX", site, other))
    return circuit


def find_example(seed, n, t_count, success_case, attempts=1000):
    rng = np.random.default_rng(seed)
    for attempt in range(1, attempts+1):
        circuit = planted_success(rng, n, t_count) if success_case else candidate(rng, n, t_count)
        result = simulate(circuit, n)
        if success_case and result.failures == 0 and any(g.kind == "CX" for g in circuit):
            return circuit, result, attempt
        if not success_case and result.failures >= 1 and result.max_bond > 1:
            return circuit, result, attempt
    raise RuntimeError("No qualifying seeded circuit found; increase attempts")


def show(name, seed, n, t_count, success_case):
    overall_started = perf_counter()
    circuit, result, attempt = find_example(seed, n, t_count, success_case)
    selection_seconds = perf_counter() - overall_started
    reference_started = perf_counter()
    dense = exact_state(circuit, n)
    candidates = [Pauli.string("I"*i + axis + "I"*(n-i-1))
                  for i in range(n) for axis in "XYZ"]
    candidates += [Pauli.string("I"*i + "X" + "I"*(j-i-1) + "X" + "I"*(n-j-1))
                   for i in range(n) for j in range(i+1, n)]
    obs = next((p for p in candidates if 0.1 < abs(exact_expectation(dense, p)) < 0.95), candidates[0])
    reference_selection_seconds = perf_counter() - reference_started
    # Run with this observable as a probe to show tensor contractions even
    # at steps where the OFD pivot condition fails.
    result = simulate(circuit, n, probe=obs)
    exact_started = perf_counter()
    expected = exact_expectation(dense, obs)
    exact_seconds = perf_counter() - exact_started
    expectation_started = perf_counter()
    observed = result.expectation(obs)
    expectation_seconds = perf_counter() - expectation_started
    assert abs(expected-observed) < 1e-9
    print("\n" + "="*76)
    print(f" {name.upper()}  |  {n} qubits, {t_count} T gates")
    print("="*76)
    print("\nCIRCUIT (left to right)")
    print("  " + "  ".join(f"{g.kind}({g.a},{g.b})" if g.b != -1 else f"{g.kind}({g.a})" for g in circuit))
    print("\nTWISTED PAULIS (before OFD)")
    for i, p in enumerate(move_t_gates(circuit, n)[1], 1):
        print(f"  T{i:<2}  {p.letters}  sign {p.sign:+d}")
    print("\nOFD / MPS UPDATES")
    for i, ((status, pauli, bond), seconds) in enumerate(zip(result.history, result.step_timings), 1):
        print(f"  T{i:<2}  {status:<7}  {pauli}  bond {bond:<2}  {seconds*1000:>8.3f} ms")
    print(f"\n  Total: {result.successes} successful OFD steps, {result.failures} failed steps")
    print(f"  GF(2) nullity: {result.initial_nullity}  |  final maximum bond: {result.max_bond}")
    print("\nFINAL EXPECTATION VALUE")
    print(f"  <{obs.letters}>  CAMPS {observed.real:.12f}  |  exact {expected.real:.12f}")
    print(f"  Absolute error: {abs(expected-observed):.2e}")
    if result.failure_expectations:
        print(f"\nMPS CONTRACTIONS AFTER FAILED OFD STEPS  (<{obs.letters}>)")
        for step, value in result.failure_expectations:
            print(f"  After T{step:<2}  value {value.real:>15.12f}  |  bond {result.history[step-1][2]}")
        print("  Intermediate state: full Clifford frame, T1 through Tk applied.")
    print("\nHOW THE CALCULATION WORKS")
    print("  1. Start from |0...0>; move each T through its preceding Cliffords.")
    print("  2. Encode the resulting Pauli strings as X/Z bits and compute GF(2) nullity.")
    for i, (status, pauli, bond) in enumerate(result.history, 1):
        action = ("use a free |0> pivot, disentangle with controlled Paulis"
                  if status == "success" else "no free X/Y pivot; add two MPS branches and compress")
        print(f"  3.{i} T{i} ({pauli}): {action}; bond {bond}.")
    print(f"  4. Conjugate {obs.letters} by C† and contract it with the MPS tensors.")
    print(f"  5. Check against a dense {n}-qubit calculation (verification only).")
    filename = "ofd_success.png" if success_case else "ofd_failure.png"
    figure_started = perf_counter()
    output = draw_circuit(circuit, n, name, result.history, "examples/output/"+filename)
    figure_seconds = perf_counter() - figure_started
    end = perf_counter()
    print("\nTIMINGS (one run; timings vary with machine and notebook load)")
    timing_rows = [
        ("Select qualifying seeded circuit (includes trial simulations)", selection_seconds),
        ("Dense reference and observable selection (demonstration only)", reference_selection_seconds),
        ("Binary Pauli propagation", result.timings["pauli_propagation"]),
        ("GF(2) nullity", result.timings["gf2_nullity"]),
        ("OFD successful steps", result.timings["ofd_success_steps"]),
        ("Failed steps: MPS sum and SVD", result.timings["failed_mps_steps"]),
        ("Intermediate failed-step expectation contractions", result.timings["failed_step_probes"]),
        ("Final CAMPS expectation contraction", expectation_seconds),
        ("Exact expectation (verification only)", exact_seconds),
        ("Draw and save figure", figure_seconds),
    ]
    for label, seconds in timing_rows:
        print(f"  {label:<59} {seconds*1000:>9.3f} ms")
    print(f"  {'Core CAMPS simulation (sum of stages)':<59} {result.timings['total_simulation']*1000:>9.3f} ms")
    print(f"  {'Entire example (includes printing and figure)':<59} {(end-overall_started)*1000:>9.3f} ms")
    print(f"\n  Figure saved to: {output}\n")


def benchmark_preprocessing():
    """Time only binary T propagation; dense 2**n vectors are never created."""
    print("\n" + "="*76)
    print(" BINARY PROPAGATION SCALING (no dense state or MPS constructed)")
    print("="*76)
    print(f"  {'Qubits':>8}  {'Cliffords':>10}  {'T gates':>8}  {'Time (ms)':>12}")
    for n in (16, 32, 64, 128):
        circuit = candidate(np.random.default_rng(300+n), n, n)
        start = perf_counter()
        _, twisted = move_t_gates(circuit, n)
        duration = perf_counter() - start
        print(f"  {n:>8}  {sum(g.kind != 'T' for g in circuit):>10}  {len(twisted):>8}  {duration*1000:>12.3f}")
    print("\n  Measures polynomial binary preprocessing only; arbitrary MPS runs may grow exponentially.\n")


if __name__ == "__main__":
    show("OFD succeeds", seed=73, n=6, t_count=3, success_case=True)
    show("OFD fails and bonds grow", seed=119, n=6, t_count=8, success_case=False)
    benchmark_preprocessing()
