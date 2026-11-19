"""Online CAMPS: apply Cliffords, Pauli rotations and Z measurements in time order.

The batch simulator in camps/simulation.py pushes every Clifford to the end of
the circuit first. That is impossible once there are mid-circuit measurements,
because what happens after a measurement depends on its outcome. This module
keeps the CAMPS state

    |psi> = C |phi>,   C = Clifford frame,   |phi> = MPS

up to date after every operation instead.

Frame bookkeeping
-----------------
`CliffordFrame` stores the Heisenberg images  C^dagger X_i C  and  C^dagger Z_i C
for every qubit i (2n Pauli strings). Any physical Pauli Q is then pulled back
to the MPS side by multiplying images:  C^dagger Q C.
  * physical Clifford G (applied to the state):  C -> G C        (left_multiply)
  * Clifford absorbed from the MPS side:          C -> C W        (right_multiply)

Operations
----------
rotation(q, axis, theta)  = exp(-i theta sigma_q / 2) = cos(theta/2) I - i sin(theta/2) sigma_q
    Twisted Pauli P = C^dagger sigma_q C. If P has X or Y on a free qubit j
    (j still |0> in the MPS) the OFD disentangler D = prod CP(j -> k) gives
    (aI + bP)|phi> = D^dagger (aI + b s sigma_j)|phi>: one single-site gate on
    the MPS plus D^dagger absorbed into the frame. Otherwise (aI + bP) is
    applied to the MPS exactly (direct sum + SVD); the bond dimension can grow.

measure_z(q)
    Twisted Pauli P = C^dagger Z_q C; outcome m has probability <phi|(1+mP)/2|phi>.
    * P has X/Y on a free qubit j: probability exactly 1/2, and the projector is
      D^dagger (1 + m s sigma_j)/2 D. The collapsed site j is a stabilizer state
      (|+>, |->, |+i> or |-i>), so it is written as V|0> with V Clifford and V is
      absorbed into the frame too. The MPS does not change at all and qubit j
      stays free.
    * otherwise: exact projector (1 + mP)/2 applied to the MPS, then renormalised.
      If one outcome has probability ~1 the state is left unchanged.

Global phases are dropped throughout.
"""
from dataclasses import dataclass, field

import numpy as np

from mps import MPS
from mps.state import MATRICES
from stabilizers_pbc import Gate, Pauli, conjugate

# Clifford V with V|0> = eigenstate of (axis) with eigenvalue e, as a product
# V = g[0] g[1] ... (applied right to left: the last gate acts first).
_PREPARE = {
    ("X", +1): [Gate("H", 0)],                                   # |+>  = H|0>
    ("X", -1): [Gate("S", 0), Gate("S", 0), Gate("H", 0)],       # |->  = Z H|0>
    ("Y", +1): [Gate("S", 0), Gate("H", 0)],                     # |+i> = S H|0>
    ("Y", -1): [Gate("SDG", 0), Gate("H", 0)],                   # |-i> = S^dag H|0>
}


def _on(gates, qubit):
    return [Gate(g.kind, qubit, g.b, g.axis) for g in gates]


def _identity(n, phase=0):
    return Pauli(np.zeros(n, np.uint8), np.zeros(n, np.uint8), phase)


def single_pauli(n, qubit, letter, sign=1):
    return Pauli.string("I" * qubit + letter + "I" * (n - qubit - 1), sign)


def _fast_pauli(x, z, phase):
    """Build a Pauli without the validation copy in Pauli.__post_init__."""
    p = object.__new__(Pauli)
    p.x, p.z, p.phase = x.astype(np.uint8), z.astype(np.uint8), int(phase) % 4
    return p


def _conj_rows(X, Z, P, gate, inverse):
    """In place: replace every row Pauli R = i^P X^X Z^Z by G R G^dag
    (or G^dag R G if inverse). Same conventions as stabilizers_pbc.conjugate."""
    def single(kind, a):
        x, z = X[:, a].astype(np.int64), Z[:, a].astype(np.int64)
        if kind == "H":
            P[:] = (P + 2 * x * z) % 4
            X[:, a], Z[:, a] = z, x
        elif kind == "S":
            P[:] = (P + x) % 4
            Z[:, a] ^= X[:, a]
        elif kind == "SDG":
            P[:] = (P - x) % 4
            Z[:, a] ^= X[:, a]
        else:
            raise ValueError(kind)

    def cx(a, b):
        xz = (X.astype(np.int64) * Z).sum(1)
        sign = ((P - xz) // 2) % 2
        xa, xb, za, zb = (X[:, a].astype(np.int64), X[:, b].astype(np.int64),
                          Z[:, a].astype(np.int64), Z[:, b].astype(np.int64))
        sign ^= xa & zb & (xb ^ za ^ 1)
        X[:, b] ^= X[:, a]
        Z[:, a] ^= Z[:, b]
        P[:] = ((X.astype(np.int64) * Z).sum(1) + 2 * sign) % 4

    k = gate.kind
    if k in ("H", "S", "SDG"):
        if inverse and k in ("S", "SDG"):
            k = "SDG" if k == "S" else "S"
        single(k, gate.a)
    elif k == "CX":
        cx(gate.a, gate.b)
    elif k == "CP":
        if gate.axis == "X":
            cx(gate.a, gate.b)
        elif gate.axis == "Z":
            single("H", gate.b); cx(gate.a, gate.b); single("H", gate.b)
        elif gate.axis == "Y":
            single("SDG", gate.b); cx(gate.a, gate.b); single("S", gate.b)
        else:
            raise ValueError("CP axis must be X, Y, or Z")
    else:
        raise ValueError("unsupported Clifford gate: " + k)


def clifford_block(gates, n):
    """Precompute U^dag P U for all 2n generators P (rows: X_0..X_{n-1}, Z_0..Z_{n-1})
    of the Clifford U = gates applied in list order. Reuse it with
    CliffordFrame.left_multiply_block to apply U in one step."""
    X = np.vstack([np.eye(n, dtype=np.uint8), np.zeros((n, n), np.uint8)])
    Z = np.vstack([np.zeros((n, n), np.uint8), np.eye(n, dtype=np.uint8)])
    P = np.zeros(2 * n, np.int64)
    for g in reversed(gates):                      # U^dag P U = g1^dag ... gm^dag P gm ... g1
        _conj_rows(X, Z, P, g, inverse=True)
    return X, Z, P


class CliffordFrame:
    """Images C^dagger X_i C (rows 0..n-1) and C^dagger Z_i C (rows n..2n-1)
    of the frame Clifford C, stored as bit arrays plus phases (i^P X^x Z^z)."""

    def __init__(self, n):
        self.n = n
        self.X = np.vstack([np.eye(n, dtype=np.uint8), np.zeros((n, n), np.uint8)])
        self.Z = np.vstack([np.zeros((n, n), np.uint8), np.eye(n, dtype=np.uint8)])
        self.P = np.zeros(2 * n, np.int64)

    def copy(self):
        f = CliffordFrame.__new__(CliffordFrame)
        f.n, f.X, f.Z, f.P = self.n, self.X.copy(), self.Z.copy(), self.P.copy()
        return f

    def _product(self, sel_x, sel_z, phase):
        """Ordered product of selected image rows (X images first, then Z images)."""
        rows = np.concatenate([np.flatnonzero(sel_x), self.n + np.flatnonzero(sel_z)])
        if len(rows) == 0:
            return np.zeros(self.n, np.uint8), np.zeros(self.n, np.uint8), phase % 4
        Rx, Rz = self.X[rows].astype(np.int64), self.Z[rows].astype(np.int64)
        before = np.cumsum(Rz, 0) - Rz                 # z of all earlier factors
        cross = int((before * Rx).sum()) % 2           # sign from moving Z past X
        ph = (phase + int(self.P[rows].sum()) + 2 * cross) % 4
        return Rx.sum(0) % 2, Rz.sum(0) % 2, ph

    def pull(self, q):
        """Return C^dagger Q C for a physical Pauli Q = i^phase X^x Z^z."""
        x, z, ph = self._product(q.x, q.z, q.phase)
        return _fast_pauli(x, z, ph)

    def left_multiply_block(self, block):
        """C -> U C, where block = clifford_block(gates of U, n)."""
        BX, BZ, BP = block
        S = np.hstack([BX, BZ]).astype(np.int64)        # which old image rows each new row uses
        A_x = self.X.astype(np.int64); A_z = self.Z.astype(np.int64)
        T = np.triu(A_z @ A_x.T, k=1)                   # z_a . x_b for a < b
        cross = ((S @ T) * S).sum(1) % 2
        newP = (BP + S @ self.P + 2 * cross) % 4
        self.X = ((S @ A_x) % 2).astype(np.uint8)
        self.Z = ((S @ A_z) % 2).astype(np.uint8)
        self.P = newP

    def left_multiply(self, gate):
        """C -> G C for one physical Clifford gate G."""
        self.left_multiply_block(clifford_block([gate], self.n))

    def right_multiply(self, gates):
        """C -> C W with W = gates[0] gates[1] ... (a product of Cliffords)."""
        for g in gates:
            _conj_rows(self.X, self.Z, self.P, g, inverse=True)   # W^dag P W


@dataclass
class OnlineCAMPS:
    """CAMPS state C|phi> updated operation by operation."""
    n: int
    frame: CliffordFrame = None
    state: MPS = None
    free: np.ndarray = None
    counts: dict = field(default_factory=lambda: {
        "rot_ofd": 0, "rot_mps": 0, "meas_ofd": 0, "meas_mps": 0, "meas_deterministic": 0})
    max_bond_seen: int = 1
    record: bool = False
    physical_gates: list = None   # for dense reconstruction in tests (record=True)
    virtual_gates: list = None

    def __post_init__(self):
        if self.frame is None:
            self.frame = CliffordFrame(self.n)
            self.state = MPS.zero(self.n)
            self.free = np.ones(self.n, dtype=bool)
        if self.record and self.physical_gates is None:
            self.physical_gates, self.virtual_gates = [], []

    def copy(self):
        c = OnlineCAMPS(self.n, self.frame.copy(), self.state.copy(), self.free.copy(),
                        dict(self.counts), self.max_bond_seen, self.record,
                        list(self.physical_gates) if self.record else None,
                        list(self.virtual_gates) if self.record else None)
        return c

    # ------------------------------------------------------------ helpers
    @property
    def max_bond(self):
        return max(self.state.bonds, default=1)

    def _note_bond(self):
        self.max_bond_seen = max(self.max_bond_seen, self.max_bond)

    def _pivot(self, p):
        hits = np.flatnonzero(self.free & (p.x == 1))
        return int(hits[0]) if len(hits) else None

    def _absorb(self, gates):
        self.frame.right_multiply(gates)
        if self.record:
            self.virtual_gates.extend(gates)

    def _disentangler(self, p, pivot):
        return [Gate("CP", pivot, j, letter) for j, letter in enumerate(p.letters)
                if j != pivot and letter != "I"]

    def _normalise(self):
        norm = np.sqrt(abs(self.state.overlap(self.state)))
        self.state.a[0] /= norm

    # ------------------------------------------------------------ operations
    def clifford(self, gate):
        """Physical Clifford gate (H, S, SDG, CX, CP): only the frame changes."""
        self.frame.left_multiply(gate)
        if self.record:
            self.physical_gates.append(gate)

    def clifford_block(self, block, gates=None):
        """Apply a whole precomputed Clifford U (see clifford_block()) in one step."""
        self.frame.left_multiply_block(block)
        if self.record:
            self.physical_gates.extend(gates)

    def rotation(self, qubit, axis, theta):
        """exp(-i theta sigma/2) on a physical qubit, sigma in {X, Y, Z}."""
        alpha, beta = np.cos(theta / 2), -1j * np.sin(theta / 2)
        p = self.frame.pull(single_pauli(self.n, qubit, axis))
        pivot = self._pivot(p)
        if pivot is None:
            self.state = self.state.rotation(p, alpha, beta)
            self.counts["rot_mps"] += 1
        else:
            local = alpha * np.eye(2) + beta * p.sign * MATRICES[p.letters[pivot]]
            self.state.local(pivot, local)
            self._absorb(self._disentangler(p, pivot))
            self.free[pivot] = False
            self.counts["rot_ofd"] += 1
        self._note_bond()

    def z_probability(self, qubit):
        """Probability of outcome +1 for a Z measurement (no collapse)."""
        p = self.frame.pull(single_pauli(self.n, qubit, "Z"))
        if self._pivot(p) is not None:
            return 0.5
        return float(np.clip((1 + self.state.expectation(p).real) / 2, 0.0, 1.0))

    def measure_z(self, qubit, rng=None, outcome=None):
        """Measure Z on a physical qubit and collapse.

        Either sample with `rng` or force `outcome` (+1/-1). Returns
        (outcome, probability of that outcome)."""
        p = self.frame.pull(single_pauli(self.n, qubit, "Z"))
        pivot = self._pivot(p)
        if pivot is not None:                      # cheap OFD measurement
            m = outcome if outcome is not None else (1 if rng.random() < 0.5 else -1)
            eigen = m * p.sign                     # eigenvalue of sigma_pivot selected
            gates = self._disentangler(p, pivot) + _on(_PREPARE[(p.letters[pivot], eigen)], pivot)
            self._absorb(gates)
            self.counts["meas_ofd"] += 1
            return m, 0.5
        p_plus = float(np.clip((1 + self.state.expectation(p).real) / 2, 0.0, 1.0))
        if outcome is None:
            m = 1 if rng.random() < p_plus else -1
        else:
            m = outcome
        prob = p_plus if m == 1 else 1 - p_plus
        if prob < 1e-12:
            raise ValueError("forced outcome has zero probability")
        if prob > 1 - 1e-10:                        # already an eigenstate
            self.counts["meas_deterministic"] += 1
            return m, 1.0
        scale = 1 / (2 * np.sqrt(prob))
        self.state = self.state.rotation(p, scale, m * scale)   # (1 + mP)/2 / sqrt(prob)
        self._normalise()
        self.counts["meas_mps"] += 1
        self._note_bond()
        return m, prob

    def expectation(self, q):
        """<psi| Q |psi> for a physical Pauli Q."""
        return self.state.expectation(self.frame.pull(q))

    def dense_state(self):
        """Full state vector C|phi> (small n, record=True only)."""
        if not self.record:
            raise RuntimeError("create with record=True to rebuild the dense state")
        from camps.reference import dense_gate
        v = self.state.vector()
        for g in reversed(self.virtual_gates):     # C = (physical) (virtual product)
            v = dense_gate(v, g, self.n)
        for g in self.physical_gates:
            v = dense_gate(v, g, self.n)
        return v
