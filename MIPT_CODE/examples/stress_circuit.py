"""Bounded stress example: 12 qubits, 45 T gates, and actual MPS bond growth.

From the repository root: python -m examples.stress_circuit
In Jupyter: %run -m examples.stress_circuit
"""
from pathlib import Path
from statistics import median
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from stabilizers_pbc import Gate, Pauli
from camps import simulate
from camps.reference import exact_state, exact_expectation


N = 12
INITIAL_T = 10
ADDITIONAL_T = 35
SEED = 2027


def build_stress_circuit(seed=SEED):
    """Seeded random Clifford layers after 10 fresh H/T pivots.

Two qubits are deliberately left idle. The remaining ten are consumed
before the random portion, so subsequent T gates exercise the MPS branch.
The circuit is reproducible; there is no runtime search for a lucky sample.
"""
    rng = np.random.default_rng(seed)
    sites = [int(i) for i in rng.permutation(N)[:INITIAL_T]]
    circuit = []
    for site in sites:
        circuit.extend([Gate("H", site), Gate("T", site)])
    baseline = list(circuit)
    for _ in range(ADDITIONAL_T):
        for _ in range(2):
            kind = str(rng.choice(["H", "S", "CX"], p=[.30, .20, .50]))
            a = int(rng.choice(sites))
            b = int(rng.choice([i for i in sites if i != a]))
            circuit.append(Gate(kind, a, b) if kind == "CX" else Gate(kind, a))
        circuit.append(Gate("T", int(rng.choice(sites))))
    return baseline, circuit, sites


def measured_simulation(circuit, repeats=5):
    """Report median single-run times; no dense reference inside timing."""
    times = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = simulate(circuit, N)
        times.append(perf_counter() - start)
    return result, median(times)


def plot_growth(result):
    steps = np.arange(1, result.t_count + 1)
    bonds = np.array([row[2] for row in result.history])
    milliseconds = np.array(result.step_timings) * 1000
    passed = np.array([row[0] == "success" for row in result.history])
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                      constrained_layout=True)
    top.step(steps, bonds, where="mid", color="#205c98", linewidth=2)
    top.scatter(steps[passed], bonds[passed], color="#238b60", label="OFD pivot found")
    top.scatter(steps[~passed], bonds[~passed], color="#b45539", label="No OFD pivot")
    top.set(ylabel="Residual MPS maximum bond", title=f"12-qubit CAMPS: {result.t_count} T gates")
    top.legend(loc="upper left", fontsize=9)
    top.grid(alpha=.2)
    bottom.bar(steps, milliseconds, color=np.where(passed, "#238b60", "#b45539"), width=.8)
    bottom.set(xlabel="T gate number", ylabel="Update time (ms)", xlim=(0, result.t_count+1))
    bottom.grid(axis="y", alpha=.2)
    output = Path("examples/output/stress_bond_timing.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.show()
    plt.close(fig)
    return output


def main():
    baseline, circuit, sites = build_stress_circuit()
    baseline_result, baseline_time = measured_simulation(baseline)
    result, full_time = measured_simulation(circuit)
    if result.max_bond < 8:
        raise AssertionError("The seeded stress example no longer shows substantial bond growth")

    # A single nontrivial Pauli observable, evaluated independently by CAMPS
    # and a small exact 2**12 reference. Exact work is outside the timings.
    # A fixed, nontrivial observable for this seed: <X_9> = 0.5.
    observable = Pauli.string("I" * 9 + "X" + "I" * (N-10))
    start = perf_counter()
    value = result.expectation(observable)
    expectation_time = perf_counter() - start
    start = perf_counter()
    dense = exact_state(circuit, N)
    reference = exact_expectation(dense, observable)
    verification_time = perf_counter() - start
    if not np.allclose(value, reference, atol=1e-9):
        raise AssertionError("CAMPS expectation differs from dense reference")

    print("\n" + "="*76)
    print(" BOUNDED STRESS DEMO  |  12 qubits, 45 T gates")
    print("="*76)
    print("\nCIRCUIT CONSTRUCTION")
    print(f"  Seed: {SEED}; first 10 H/T pairs consume sites {sites}.")
    print("  Then 35 T gates are interleaved with two seeded random H/S/CX gates each.")
    print("  Cliffords and Ts remain spread through this second circuit segment.")
    print("\nRESULTS")
    print(f"  {'Circuit':<34} {'T gates':>8} {'OFD misses':>12} {'Max bond':>10}")
    print(f"  {'First 10 T gates only':<34} {baseline_result.t_count:>8} {baseline_result.failures:>12} {baseline_result.max_bond:>10}")
    print(f"  {'Full seeded circuit':<34} {result.t_count:>8} {result.failures:>12} {result.max_bond:>10}")
    print(f"\n  <{observable.letters}> CAMPS: {value.real:.12f}  |  exact: {reference.real:.12f}")
    print(f"  Absolute error: {abs(value-reference):.2e}")
    print("\nTIMINGS (five independent runs, median; excludes exact check and plotting)")
    print(f"  {'First 10 T gates: core simulation':<48} {baseline_time*1000:>10.3f} ms")
    print(f"  {'Full 45 T gates: core simulation':<48} {full_time*1000:>10.3f} ms")
    print(f"  {'Runtime ratio, full / first 10':<48} {full_time/baseline_time:>10.2f} x")
    print("\n  Full-circuit stage timings below are from the last run, not medians:")
    print(f"  {'Full circuit, binary propagation':<48} {result.timings['pauli_propagation']*1000:>10.3f} ms")
    print(f"  {'Full circuit, GF(2) nullity':<48} {result.timings['gf2_nullity']*1000:>10.3f} ms")
    print(f"  {'Full circuit, successful OFD updates':<48} {result.timings['ofd_success_steps']*1000:>10.3f} ms")
    print(f"  {'Full circuit, failed-step MPS sums and SVDs':<48} {result.timings['failed_mps_steps']*1000:>10.3f} ms")
    print(f"  {'Final MPS expectation contraction':<48} {expectation_time*1000:>10.3f} ms")
    print(f"  {'Exact 12-qubit verification (separate)':<48} {verification_time*1000:>10.3f} ms")
    print("\n  The full/initial ratio includes both extra gates and bond growth;")
    print("  it does not isolate the cost of bond growth by itself.")
    print("  A larger T count is possible, but no universal safe maximum exists.")
    output = plot_growth(result)
    print(f"\n  Growth plot saved to: {output}\n")


if __name__ == "__main__":
    main()
