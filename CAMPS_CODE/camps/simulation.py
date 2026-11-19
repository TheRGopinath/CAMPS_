"""Convention: final state = C |MPS>; circuit gates listed in time order.

T = exp(i*pi/8) (cos(pi/8) I - i sin(pi/8) Z). Global phase is omitted.
No MPS bond cap is imposed: bad OFD instances can still cost exponentially.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from stabilizers_pbc import Pauli, Gate, Tableau, conjugate, gf2_nullity
from mps import MPS

ALPHA = np.cos(np.pi/8)
BETA = -1j*np.sin(np.pi/8)


def _through(pauli, gates, inverse=False):
    """C P C† for application-order list; inverse gives C† P C."""
    result = pauli.copy()
    for gate in (reversed(gates) if inverse else gates):
        result = conjugate(result, gate, inverse=inverse)
    return result


def move_t_gates(circuit, n):
    """U = C R_t ... R_1: return C and Pauli generators of R in time order.

Process each T on arrival using C† Z C. Clifford gates never touch the
MPS during this pass; each Pauli update visits O(n) binary labels.
"""
    frame = []
    twisted = []
    for gate in circuit:
        if not (0 <= gate.a < n) or (gate.b != -1 and not 0 <= gate.b < n):
            raise ValueError("gate qubit index out of range")
        if gate.kind == "T":
            twisted.append(_through(Pauli.string("I"*gate.a+"Z"+"I"*(n-gate.a-1)),
                                    frame, inverse=True))
        else:
            if gate.kind not in ("H", "S", "SDG", "CX", "CP"):
                raise ValueError("unknown gate: " + gate.kind)
            frame.append(gate)
    return frame, twisted


def ofd_nullity(twisted):
    """Left nullity of binary matrix: I/Z -> 0, X/Y -> 1."""
    if not twisted:
        return 0
    return gf2_nullity(np.stack([p.x for p in twisted]))


@dataclass
class CAMPS:
    frame: list
    state: MPS
    successes: int
    failures: int
    initial_nullity: int
    history: list
    failure_expectations: list
    t_count: int
    timings: dict
    step_timings: list

    def expectation(self, observable):
        """Use tableau for Clifford-only states; otherwise local MPS contraction."""
        if self.t_count == 0:
            tableau = Tableau(len(self.state.a))
            for gate in self.frame:
                tableau.apply(gate)
            return complex(tableau.expectation(observable))
        return self.state.expectation(_through(observable, self.frame, inverse=True))

    @property
    def max_bond(self):
        return max(self.state.bonds, default=1)


def simulate(circuit, n, probe=None):
    """Run OFD. Every successful pivot consumes a free |0> qubit.

For unsuccessful pivots use an exact MPS direct sum and SVD. This may grow
exponentially in the number of failures. Recalculate future strings by
conjugating them through each constructed controlled-Pauli disentangler.

If probe is supplied, also contract <probe> immediately after every failed
OFD step. These intermediate values refer to C times the partially evolved
MPS (C contains all Clifford gates); only the last one is the full circuit.
"""
    if probe is not None and len(probe.x) != n:
        raise ValueError("probe must act on all n qubits")
    started = perf_counter()
    frame, paulis = move_t_gates(circuit, n)
    after_propagation = perf_counter()
    nullity = ofd_nullity(paulis)
    after_nullity = perf_counter()
    free = np.ones(n, dtype=bool)
    state = MPS.zero(n)
    successes = failures = 0
    history = []
    failure_expectations = []
    step_timings = []
    probe_seconds = 0.0
    for k, p in enumerate(paulis):
        step_started = perf_counter()
        pivot = next((i for i in range(n) if free[i] and p.x[i]), None)
        if pivot is None:
            state = state.rotation(p, ALPHA, BETA)
            failures += 1
            history.append(("failed", p.letters, max(state.bonds, default=1)))
            update_done = perf_counter()
            if probe is not None:
                rotated_probe = _through(probe, frame, inverse=True)
                value = state.expectation(rotated_probe)  # local tensor contraction
                failure_expectations.append((k+1, value))
                probe_seconds += perf_counter() - update_done
            step_timings.append(perf_counter() - step_started)
            continue
        # |0> at pivot and X/Y there: D(alpha I+beta P)|psi> has
        # only the pivot rotated. CP gates on other sites implement D.
        axis = p.letters[pivot]
        local = Pauli.string("I"*pivot + axis + "I"*(n-pivot-1), p.sign)
        state = state.rotation(local, ALPHA, BETA)
        disentangler = [Gate("CP", pivot, j, letter) for j, letter in enumerate(p.letters)
                         if j != pivot and letter != "I"]
        # Cnew = C D†; application-order list therefore prepends D†.
        frame = list(reversed(disentangler)) + frame
        for j in range(k+1, len(paulis)):
            paulis[j] = _through(paulis[j], disentangler)
        free[pivot] = False
        successes += 1
        history.append(("success", p.letters, max(state.bonds, default=1)))
        step_timings.append(perf_counter() - step_started)
    finished = perf_counter()
    timings = {
        "pauli_propagation": after_propagation - started,
        "gf2_nullity": after_nullity - after_propagation,
        "ofd_success_steps": sum(dt for dt, h in zip(step_timings, history) if h[0] == "success"),
        "failed_mps_steps": sum(dt for dt, h in zip(step_timings, history) if h[0] == "failed") - probe_seconds,
        "failed_step_probes": probe_seconds,
        "total_simulation": finished - started,
    }
    return CAMPS(frame, state, successes, failures, nullity, history,
                 failure_expectations, len(paulis), timings, step_timings)
