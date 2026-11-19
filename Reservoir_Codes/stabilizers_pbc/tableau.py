"""Stabilizer/destabilizer tableau with arbitrary Hermitian Pauli measurement."""
import numpy as np
from .pauli import Pauli, conjugate


class Tableau:
    def __init__(self, n):
        if n < 1:
            raise ValueError("at least one qubit required")
        self.n = n
        self.stabilizers = [Pauli.string("I" * i + "Z" + "I" * (n-i-1)) for i in range(n)]
        self.destabilizers = [Pauli.string("I" * i + "X" + "I" * (n-i-1)) for i in range(n)]

    def apply(self, gate):
        self.stabilizers = [conjugate(p, gate) for p in self.stabilizers]
        self.destabilizers = [conjugate(p, gate) for p in self.destabilizers]

    def expectation(self, observable):
        """Exact <P>: 0 when P anticommutes with a generator; otherwise +/-1.

        Does not measure or alter the tableau. Intended for a pure stabilizer
        state and a Hermitian Pauli observable.
        """
        if len(observable.x) != self.n:
            raise ValueError("observable qubit count differs from tableau")
        _ = observable.sign  # reject non-Hermitian operators
        if any(s.anticommutes(observable) for s in self.stabilizers):
            return 0
        basis = {}
        for s in self.stabilizers:
            row = s.copy()
            for col in sorted(basis):
                if row.binary()[col]:
                    row = row @ basis[col]
            bits = np.flatnonzero(row.binary())
            if len(bits):
                basis[int(bits[0])] = row
        remaining = observable.copy()
        for col in sorted(basis):
            if remaining.binary()[col]:
                remaining = remaining @ basis[col]
        if np.any(remaining.binary()):
            raise RuntimeError("commuting Pauli was not found in pure-state stabilizer span")
        return remaining.sign

    def measure(self, observable, rng=None):
        """Return (outcome +/-1, probability); update the tableau in place."""
        if len(observable.x) != self.n or observable.sign not in (-1, 1):
            raise ValueError("expected Hermitian n-qubit Pauli")
        anti = [i for i, s in enumerate(self.stabilizers) if s.anticommutes(observable)]
        if anti:
            pivot = anti[0]
            old = self.stabilizers[pivot].copy()
            for i in anti[1:]:
                self.stabilizers[i] = self.stabilizers[i] @ old
            for i, d in enumerate(self.destabilizers):
                if i != pivot and d.anticommutes(observable):
                    self.destabilizers[i] = d @ old
            self.destabilizers[pivot] = old
            rng = np.random.default_rng() if rng is None else rng
            outcome = 1 if rng.integers(2) == 0 else -1
            measured = observable.copy()
            measured.phase = (measured.phase + (2 if outcome == -1 else 0)) % 4
            self.stabilizers[pivot] = measured
            return outcome, 0.5

        return self.expectation(observable), 1.0
