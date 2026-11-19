import numpy as np

MATRICES = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1., -1.]).astype(complex),
}


class MPS:
    """Tensors A[site][left_bond, physical_bit, right_bond].

    No truncation is used in CAMPS examples. Optional compression uses the
    stated discarded squared singular-value weight; no hidden renormalization.
    """
    def __init__(self, tensors):
        self.a = [np.asarray(t, dtype=complex).copy() for t in tensors]
        if not self.a or any(t.ndim != 3 or t.shape[1] != 2 for t in self.a):
            raise ValueError("expected nonempty rank-three qubit tensors")
        if self.a[0].shape[0] != 1 or self.a[-1].shape[2] != 1:
            raise ValueError("boundary bond dimensions must be one")
        if any(l.shape[2] != r.shape[0] for l, r in zip(self.a, self.a[1:])):
            raise ValueError("neighboring bond dimensions differ")

    @classmethod
    def zero(cls, n):
        if n < 1:
            raise ValueError("at least one qubit required")
        v = np.array([1., 0.], dtype=complex)
        return cls([v.reshape(1, 2, 1) for _ in range(n)])

    @classmethod
    def from_vector(cls, vector, n, max_bond=None):
        """Sequential SVD, primarily a reference for small-system checks."""
        v = np.asarray(vector, dtype=complex).reshape(-1)
        if n < 1 or v.size != 2**n:
            raise ValueError("vector must contain 2**n amplitudes")
        tensors = []
        left = 1
        rest = v
        for _ in range(n-1):
            u, s, vh = np.linalg.svd(rest.reshape(left * 2, -1), full_matrices=False)
            keep = len(s) if max_bond is None else min(max_bond, len(s))
            tensors.append(u[:, :keep].reshape(left, 2, keep))
            rest = s[:keep, None] * vh[:keep]
            left = keep
        tensors.append(rest.reshape(left, 2, 1))
        return cls(tensors)

    def copy(self):
        return MPS(self.a)

    def vector(self):
        state = self.a[0][0]
        for tensor in self.a[1:]:
            state = np.tensordot(state, tensor, axes=(-1, 0))
            state = state.reshape(-1, state.shape[-1])
        return state[:, 0]

    @property
    def bonds(self):
        return [t.shape[2] for t in self.a[:-1]]

    def local(self, site, matrix):
        self.a[site] = np.einsum("st,atb->asb", matrix, self.a[site])
        return self

    def scaled(self, value):
        result = self.copy()
        result.a[0] *= value
        return result

    def add(self, other):
        """Exact direct-sum MPS for |self> + |other>."""
        if len(self.a) != len(other.a):
            raise ValueError("site count mismatch")
        n = len(self.a)
        if n == 1:
            return MPS([self.a[0] + other.a[0]])
        out = []
        for i, (a, b) in enumerate(zip(self.a, other.a)):
            if i == 0:
                out.append(np.concatenate((a, b), axis=2))
            elif i == n-1:
                out.append(np.concatenate((a, b), axis=0))
            else:
                c = np.zeros((a.shape[0]+b.shape[0], 2, a.shape[2]+b.shape[2]), complex)
                c[:a.shape[0], :, :a.shape[2]] = a
                c[a.shape[0]:, :, a.shape[2]:] = b
                out.append(c)
        return MPS(out)

    def compress(self, max_bond=None, cutoff=0.0):
        """Right canonicalize, then left-to-right SVD; return discarded weight."""
        discarded = 0.0
        # Without the right sweep, direct-sum blocks can retain spurious bonds:
        # the left local matrix may have rank 2 even when the full state is product.
        for i in range(len(self.a)-1, 0, -1):
            left, physical, right = self.a[i].shape
            u, s, vh = np.linalg.svd(self.a[i].reshape(left, physical*right), full_matrices=False)
            self.a[i] = vh.reshape(len(s), physical, right)
            self.a[i-1] = np.tensordot(self.a[i-1], u*s, axes=(2, 0))
        for i in range(len(self.a)-1):
            left, physical, right = self.a[i].shape
            u, s, vh = np.linalg.svd(self.a[i].reshape(left*physical, right), full_matrices=False)
            keep = int(np.count_nonzero(s > cutoff))
            if max_bond is not None:
                keep = min(keep, max_bond)
            keep = max(1, keep)
            discarded += float(np.sum(s[keep:]**2))
            self.a[i] = u[:, :keep].reshape(left, physical, keep)
            self.a[i+1] = np.tensordot(s[:keep, None]*vh[:keep], self.a[i+1], axes=(1, 0))
        return discarded

    def pauli(self, p):
        result = self.copy()
        for site, letter in enumerate(p.letters):
            result.local(site, MATRICES[letter])
        result.a[0] *= (1j)**p.phase / (1j)**sum(c == "Y" for c in p.letters)
        return result

    def rotation(self, p, alpha, beta):
        """Exact (alpha I + beta P)|psi>, then remove zero singular values."""
        result = self.scaled(alpha).add(self.pauli(p).scaled(beta))
        result.compress(cutoff=1e-12)
        return result

    def overlap(self, other):
        env = np.ones((1, 1), complex)
        for a, b in zip(self.a, other.a):
            env = np.einsum("ab,asc,bsd->cd", env, a.conj(), b)
        return env[0, 0]

    def expectation(self, p):
        """Contract <MPS|P|MPS> site by site, without building 2**n amplitudes."""
        return self.overlap(self.pauli(p))
