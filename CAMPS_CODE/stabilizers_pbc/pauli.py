"""P = i**phase X**x Z**z (X factors before Z factors, qubits 0..n-1).

Y=iXZ. Phase is mod 4; Hermitian Paulis have phase-sum(x*z) even.
Gate lists are in application order. Gate('CX', control, target), and
Gate('CP', control, target, axis) are supported. CP is controlled X/Y/Z.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Gate:
    kind: str
    a: int
    b: int = -1
    axis: str = ""


@dataclass
class Pauli:
    x: np.ndarray
    z: np.ndarray
    phase: int = 0

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=np.uint8).copy()
        self.z = np.asarray(self.z, dtype=np.uint8).copy()
        if self.x.ndim != 1 or self.x.shape != self.z.shape:
            raise ValueError("x and z must have equal one-dimensional shapes")
        if np.any(self.x > 1) or np.any(self.z > 1):
            raise ValueError("Pauli labels must be bits")
        self.phase %= 4

    @classmethod
    def string(cls, letters, sign=1):
        if any(c not in "IXYZ" for c in letters) or sign not in (-1, 1):
            raise ValueError("use I/X/Y/Z and sign +/-1")
        x = np.array([c in "XY" for c in letters], dtype=np.uint8)
        z = np.array([c in "YZ" for c in letters], dtype=np.uint8)
        return cls(x, z, (sum(c == "Y" for c in letters) + (2 if sign < 0 else 0)) % 4)

    @property
    def letters(self):
        return "".join("IXZY"[int(a) + 2 * int(b)] for a, b in zip(self.x, self.z))

    @property
    def sign(self):
        difference = (self.phase - int(np.dot(self.x, self.z))) % 4
        if difference % 2:
            raise ValueError("Pauli is anti-Hermitian")
        return 1 if difference == 0 else -1

    def copy(self):
        return Pauli(self.x, self.z, self.phase)

    def __matmul__(self, other):
        if len(self.x) != len(other.x):
            raise ValueError("qubit count mismatch")
        return Pauli(self.x ^ other.x, self.z ^ other.z,
                     (self.phase + other.phase + 2 * int(np.dot(self.z, other.x))) % 4)

    def anticommutes(self, other):
        return bool((int(np.dot(self.x, other.z)) + int(np.dot(self.z, other.x))) % 2)

    def binary(self):
        return np.r_[self.x, self.z]


def _cx(p, a, b):
    # Update Hermitian sign as in Aaronson-Gottesman, then restore i**phase X**x Z**z.
    sign = (p.phase - int(np.dot(p.x, p.z))) // 2 % 2
    xa, xb, za, zb = map(int, (p.x[a], p.x[b], p.z[a], p.z[b]))
    sign ^= xa & zb & (xb ^ za ^ 1)
    p.x[b] ^= p.x[a]
    p.z[a] ^= p.z[b]
    p.phase = (int(np.dot(p.x, p.z)) + 2 * sign) % 4


def _single(p, kind, a):
    x, z = int(p.x[a]), int(p.z[a])
    if kind == "H":
        p.phase = (p.phase + 2 * x * z) % 4
        p.x[a], p.z[a] = z, x
    elif kind == "S":
        p.phase = (p.phase + x) % 4
        p.z[a] ^= p.x[a]
    elif kind == "SDG":
        # S dagger: X -> -Y, Y -> X.
        p.phase = (p.phase - x) % 4
        p.z[a] ^= p.x[a]
    else:
        raise ValueError(kind)


def conjugate(pauli, gate, inverse=False):
    """Return G P G†; if inverse=True return G† P G."""
    p = pauli.copy()
    if gate.kind in ("H", "S", "SDG"):
        kind = gate.kind
        if inverse and kind in ("S", "SDG"):
            kind = "SDG" if kind == "S" else "S"
        _single(p, kind, gate.a)
    elif gate.kind == "CX":
        _cx(p, gate.a, gate.b)
    elif gate.kind == "CP":
        # CP = (I⊗B) CX (I⊗B†), B maps X to axis.
        if gate.axis == "X":
            _cx(p, gate.a, gate.b)
        elif gate.axis == "Z":
            _single(p, "H", gate.b)
            _cx(p, gate.a, gate.b)
            _single(p, "H", gate.b)
        elif gate.axis == "Y":
            _single(p, "SDG", gate.b)
            _cx(p, gate.a, gate.b)
            _single(p, "S", gate.b)
        else:
            raise ValueError("CP axis must be X, Y, or Z")
    else:
        raise ValueError("unsupported Clifford gate: " + gate.kind)
    return p


def gf2_rank(rows):
    a = np.asarray(rows, dtype=np.uint8).copy()
    if a.ndim != 2:
        raise ValueError("expected a two-dimensional binary matrix")
    rank = 0
    for col in range(a.shape[1]):
        pivots = np.flatnonzero(a[rank:, col])
        if len(pivots) == 0:
            continue
        pivot = rank + int(pivots[0])
        a[[rank, pivot]] = a[[pivot, rank]]
        for row in range(rank + 1, len(a)):
            if a[row, col]:
                a[row] ^= a[rank]
        rank += 1
        if rank == len(a):
            break
    return rank


def gf2_nullity(rows):
    """Number of dependent rows (left nullity), as in the OFD paper."""
    return len(rows) - gf2_rank(rows)
