import numpy as np
from stabilizers_pbc import Pauli, Gate, Tableau, conjugate
from camps import simulate, move_t_gates
from camps.reference import exact_state, exact_expectation
from mps import MPS


def test_tableau():
    tab = Tableau(2)
    tab.apply(Gate("H", 0))
    tab.apply(Gate("CX", 0, 1))
    assert tab.measure(Pauli.string("ZZ")) == (1, 1.0)
    assert tab.measure(Pauli.string("XX")) == (1, 1.0)
    assert tab.expectation(Pauli.string("ZZ")) == 1
    assert tab.expectation(Pauli.string("ZZ", sign=-1)) == -1
    assert tab.expectation(Pauli.string("ZI")) == 0
    result, prob = tab.measure(Pauli.string("ZI"), np.random.default_rng(9))
    assert prob == 0.5 and tab.measure(Pauli.string("ZI")) == (result, 1.0)
    assert tab.expectation(Pauli.string("ZI")) == result


def test_mps():
    rng = np.random.default_rng(3)
    v = rng.normal(size=32) + 1j*rng.normal(size=32)
    state = MPS.from_vector(v, 5)
    assert np.allclose(state.vector(), v)
    assert np.allclose(state.rotation(Pauli.string("XYZII"), .3, .7).vector(),
                       .3*v + .7*state.pauli(Pauli.string("XYZII")).vector())


def test_random_circuits():
    rng = np.random.default_rng(21)
    for n in (2, 3, 4, 5):
        for _ in range(15):
            circuit = []
            for _ in range(12):
                kind = rng.choice(["H", "S", "SDG", "CX", "T"])
                a = int(rng.integers(n))
                b = (a+int(rng.integers(1, n))) % n
                circuit.append(Gate(kind, a, b) if kind == "CX" else Gate(kind, a))
            result = simulate(circuit, n)
            dense = exact_state(circuit, n)
            reconstructed = result.state.vector()
            for gate in result.frame:
                from camps.reference import dense_gate
                reconstructed = dense_gate(reconstructed, gate, n)
            # T global phases are omitted.
            assert np.allclose(reconstructed, dense*np.exp(-1j*np.pi/8*sum(g.kind == "T" for g in circuit)), atol=1e-9)
            for letters in ("Z"*n, "X"+"I"*(n-1)):
                p = Pauli.string(letters)
                assert np.allclose(result.expectation(p), exact_expectation(dense, p), atol=1e-9)


def test_pauli_gate_conjugation():
    from camps.reference import dense_gate
    n = 2
    for gate in (Gate("H", 0), Gate("S", 1), Gate("SDG", 0), Gate("CX", 0, 1),
                 Gate("CP", 0, 1, "X"), Gate("CP", 0, 1, "Y"), Gate("CP", 0, 1, "Z")):
        for letters in ("XI", "YI", "IZ", "XY", "YY"):
            p = Pauli.string(letters)
            basis = MPS.from_vector(np.array([1., 2j, 3., 4j]), n)
            evolved = dense_gate(basis.pauli(p).vector(), gate, n)
            transformed = MPS.from_vector(dense_gate(basis.vector(), gate, n), n).pauli(conjugate(p, gate)).vector()
            assert np.allclose(evolved, transformed)


def test_expectations_with_and_without_ofd():
    from camps.reference import dense_gate
    cliffords = [Gate("H", 0), Gate("CX", 0, 1)]
    pure = simulate(cliffords, 2)
    assert pure.t_count == 0
    assert pure.expectation(Pauli.string("XX")) == 1
    assert pure.expectation(Pauli.string("ZI")) == 0
    # Consume both free qubits; a later twisted T cannot use an OFD pivot.
    circuit = [Gate("H", 0), Gate("T", 0), Gate("H", 1), Gate("T", 1),
               Gate("CX", 0, 1), Gate("T", 1)]
    probe = Pauli.string("XX")
    result = simulate(circuit, 2, probe=probe)
    assert result.failures >= 1 and len(result.failure_expectations) == result.failures
    assert np.allclose(result.expectation(probe), exact_expectation(exact_state(circuit, 2), probe))
    # Check every intermediate tensor contraction using a dense reference
    # with the final Clifford frame and only the first k twisted T rotations.
    frame, twisted = move_t_gates(circuit, 2)
    for step, value in result.failure_expectations:
        state = np.array([1, 0, 0, 0], dtype=complex)
        for p in twisted[:step]:
            state = np.cos(np.pi/8)*state - 1j*np.sin(np.pi/8)*MPS.from_vector(state, 2).pauli(p).vector()
        for gate in frame:
            state = dense_gate(state, gate, 2)
        assert np.allclose(value, exact_expectation(state, probe))
