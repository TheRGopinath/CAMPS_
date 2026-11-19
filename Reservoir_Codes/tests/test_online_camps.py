"""Check online CAMPS (Cliffords + arbitrary rotations + Z measurements)
against a brute-force state vector, step by step."""
import numpy as np

from camps.online import OnlineCAMPS, single_pauli
from camps.reference import dense_gate
from mipt import two_qubit_clifford_words
from mps.state import MATRICES

WORD_GATES = {0: ("H", 0), 1: ("H", 1), 2: ("S", 0), 3: ("S", 1), 4: ("CX", 0, 1), 5: ("CX", 1, 0)}


def word_to_gates(word, a, b):
    from stabilizers_pbc import Gate
    out = []
    for g in word:
        spec = WORD_GATES[g]
        q = (a, b)
        out.append(Gate(spec[0], q[spec[1]]) if len(spec) == 2 else Gate(spec[0], q[spec[1]], q[spec[2]]))
    return out


def dense_single(v, n, q, u):
    t = v.reshape((2,) * n)
    return np.moveaxis(np.tensordot(u, t, axes=(1, q)), 0, q).reshape(-1)


def dense_measure(v, n, q, m):
    t = v.reshape((2,) * n).copy()
    idx = [slice(None)] * n
    idx[q] = 1 if m == 1 else 0          # zero the other branch
    t[tuple(idx)] = 0
    w = t.reshape(-1)
    prob = float(np.vdot(w, w).real)
    return w / np.sqrt(prob) if prob > 0 else w, prob


def run_random(n, steps, seed):
    rng = np.random.default_rng(seed)
    words = two_qubit_clifford_words()
    sim = OnlineCAMPS(n, record=True)
    v = np.zeros(2 ** n, complex); v[0] = 1
    worst = 0.0
    for _ in range(steps):
        r = rng.random()
        if r < 0.35:                                       # random 2-qubit Clifford
            a = int(rng.integers(n - 1))
            for g in word_to_gates(words[int(rng.integers(len(words)))], a, a + 1):
                sim.clifford(g); v = dense_gate(v, g, n)
        elif r < 0.7:                                      # arbitrary-angle rotation
            q = int(rng.integers(n)); axis = "XYZ"[int(rng.integers(3))]
            theta = float(rng.uniform(0, 2 * np.pi))
            sim.rotation(q, axis, theta)
            u = np.cos(theta / 2) * np.eye(2) - 1j * np.sin(theta / 2) * MATRICES[axis]
            v = dense_single(v, n, q, u)
        else:                                              # Z measurement
            q = int(rng.integers(n))
            m, prob = sim.measure_z(q, rng)
            v, prob_dense = dense_measure(v, n, q, m)
            worst = max(worst, abs(prob - prob_dense))
    fidelity = abs(np.vdot(v, sim.dense_state())) ** 2
    return worst, fidelity, sim.counts


def test_online_camps_matches_state_vector():
    totals = {}
    for n in (2, 3, 4, 5):
        for seed in range(12):
            worst, fid, counts = run_random(n, 60, 1000 * n + seed)
            assert worst < 1e-9, (n, seed, worst)
            assert abs(fid - 1) < 1e-9, (n, seed, fid)
            for k, c in counts.items():
                totals[k] = totals.get(k, 0) + c
    # every code path must actually have been exercised
    for key in ("rot_ofd", "rot_mps", "meas_ofd", "meas_mps", "meas_deterministic"):
        assert totals[key] > 0, (key, totals)


def test_forced_outcomes_sum_to_one():
    rng = np.random.default_rng(4)
    sim = OnlineCAMPS(3)
    sim.rotation(0, "Y", 1.1); sim.rotation(1, "X", 0.4)
    from stabilizers_pbc import Gate
    sim.clifford(Gate("CX", 0, 1)); sim.clifford(Gate("H", 2)); sim.clifford(Gate("CX", 2, 1))
    total = 0.0
    for m in (1, -1):
        c = sim.copy()
        try:
            _, p = c.measure_z(1, outcome=m)
        except ValueError:
            p = 0.0
        total += p
    assert abs(total - 1) < 1e-12
    assert abs(sim.z_probability(1) - (1 + sim.expectation(single_pauli(3, 1, "Z")).real) / 2) < 1e-12


if __name__ == "__main__":
    test_online_camps_matches_state_vector()
    test_forced_outcomes_sum_to_one()
    print("online CAMPS checks passed")
