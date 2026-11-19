"""Monitored Clifford brickwork circuit, with uniform random 2-qubit Cliffords.

A full timestep is: odd bonds -> independent Z measurements -> even bonds ->
independent Z measurements. Qubit 0 is leftmost; open boundary conditions.
The quantum trajectory is pure, and measurements use Born-rule outcomes.

This module uses a compact *stabilizer-only* signed binary tableau for speed.
It is compatible with the Pauli phase convention in stabilizers_pbc/pauli.py,
but does not modify the pre-existing CAMPS implementation.
"""
from collections import deque
from functools import lru_cache
import numpy as np

# Signed Pauli encoding on two qubits: (x0,x1,z0,z1, sign), with
# P=(-1)^sign (i**(x0*z0+x1*z1)) X^x Z^z.
# To enumerate Cliffords, store images of X0,X1,Z0,Z1 as 4 signed Paulis.


def _local_conj(code, gate):
    x0 = code & 1; x1 = (code >> 1) & 1
    z0 = (code >> 2) & 1; z1 = (code >> 3) & 1
    sign = (code >> 4) & 1
    if gate == 0:  # H on 0
        sign ^= x0 & z0; x0, z0 = z0, x0
    elif gate == 1:  # H on 1
        sign ^= x1 & z1; x1, z1 = z1, x1
    elif gate == 2:  # S on 0
        sign ^= x0 & z0; z0 ^= x0
    elif gate == 3:  # S on 1
        sign ^= x1 & z1; z1 ^= x1
    elif gate == 4:  # CX 0 -> 1
        sign ^= x0 & z1 & (x1 ^ z0 ^ 1)
        x1 ^= x0; z0 ^= z1
    elif gate == 5:  # CX 1 -> 0
        sign ^= x1 & z0 & (x0 ^ z1 ^ 1)
        x0 ^= x1; z1 ^= z0
    return x0 | (x1 << 1) | (z0 << 2) | (z1 << 3) | (sign << 4)


@lru_cache(maxsize=1)
def two_qubit_clifford_words():
    """BFS-enumerate the 11,520 distinct 2-qubit Cliffords modulo global phase.

    A uniformly chosen BFS word is therefore an EXACT uniform sample from the
    projective 2-qubit Clifford group, not a fixed-length random walk.
    This one-time construction can take several seconds.
    """
    identity = (1, 2, 4, 8)  # X0, X1, Z0, Z1
    seen = {identity}
    queue = deque([(identity, ())])
    words = []
    while queue:
        state, word = queue.popleft()
        words.append(word)
        for gate in range(6):
            nxt = tuple(_local_conj(p, gate) for p in state)
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, word + (gate,)))
    if len(words) != 11520:
        raise RuntimeError(f'Expected 11520 two-qubit Cliffords; found {len(words)}')
    return tuple(words)


class FastStabilizer:
    """n independent Hermitian stabilizer generators, in binary symplectic form.

    x,z are (n,n) uint8 arrays; sign is an n-vector, where each row is
    (-1)^sign i^(sum_j x_j z_j) X^x Z^z. No statevector is ever constructed.
    """
    def __init__(self, n):
        self.n = int(n)
        self.x = np.concatenate((np.zeros((n,n),dtype=np.uint8),np.eye(n,dtype=np.uint8)),axis=0)
        self.z = np.concatenate((np.eye(n,dtype=np.uint8),np.zeros((n,n),dtype=np.uint8)),axis=0)
        self.sign = np.zeros(2*n,dtype=np.uint8)

    def h(self, q):
        self.sign ^= self.x[:, q] & self.z[:, q]
        tmp = self.x[:, q].copy()
        self.x[:, q] = self.z[:, q]
        self.z[:, q] = tmp

    def s(self, q):
        self.sign ^= self.x[:, q] & self.z[:, q]
        self.z[:, q] ^= self.x[:, q]

    def cx(self, c, t):
        xc = self.x[:, c].copy(); xt = self.x[:, t].copy()
        zc = self.z[:, c].copy(); zt = self.z[:, t].copy()
        self.sign ^= xc & zt & (xt ^ zc ^ 1)
        self.x[:, t] ^= xc
        self.z[:, c] ^= zt

    def random_two_qubit_clifford(self, a, b, rng, words):
        word = words[int(rng.integers(len(words)))]
        for gate in word:
            if gate == 0: self.h(a)
            elif gate == 1: self.h(b)
            elif gate == 2: self.s(a)
            elif gate == 3: self.s(b)
            elif gate == 4: self.cx(a, b)
            else: self.cx(b, a)
        return word

    def _multiply_rows(self, dst, src):
        """Replace row dst by product of commuting Hermitian rows dst*src."""
        # i^p X^x Z^z convention, converted to Hermitian sign convention.
        # Multiplication exponent p = x.z + 2*sign (mod 4).
        xa = self.x[dst].copy(); za = self.z[dst].copy()
        xb = self.x[src]; zb = self.z[src]
        p = (int(xa @ za) + 2*int(self.sign[dst]) +
             int(xb @ zb) + 2*int(self.sign[src]) + 2*int(za @ xb)) % 4
        xn = xa ^ xb; zn = za ^ zb
        difference = (p - int(xn @ zn)) % 4
        if difference % 2:
            raise RuntimeError('Noncommuting stabilizer rows encountered')
        self.sign[dst] = difference // 2
        self.x[dst] = xn; self.z[dst] = zn

    def z_expectation(self, q):
        """Deterministic Z expectation using the destabilizer dual basis."""
        if np.any(self.x[:self.n,q]): return 0
        # Z_q = product of precisely those stabilizers whose dual
        # destabilizers anticommute with Z_q.
        coeff = self.x[self.n:,q]
        x=np.zeros(self.n,dtype=np.uint8);z=np.zeros(self.n,dtype=np.uint8)
        phase=0
        for j in np.flatnonzero(coeff):
            phase=(phase+int(self.x[j]@self.z[j])+2*int(self.sign[j])+
                   2*int(z@self.x[j]))%4
            x ^= self.x[j];z ^= self.z[j]
        if np.any(x) or z[q]!=1 or int(z.sum())!=1:
            raise RuntimeError('Destabilizer/stabilizer duality lost')
        return 1 if phase==0 else -1

    def measure_z(self, q, rng):
        """Sample a projective Z measurement and update both tableau halves."""
        anti=np.flatnonzero(self.x[:self.n,q])
        if len(anti)==0: return self.z_expectation(q),1.0
        pivot=int(anti[0])
        # Save old stabilizer row as new destabilizer pivot.
        old_x=self.x[pivot].copy();old_z=self.z[pivot].copy()
        old_sign=int(self.sign[pivot])
        for j in anti[1:]: self._multiply_rows(int(j),pivot)
        for j in range(self.n,2*self.n):
            if j != self.n+pivot and self.x[j,q]:
                self._multiply_rows(j,pivot)
        self.x[self.n+pivot]=old_x
        self.z[self.n+pivot]=old_z
        self.sign[self.n+pivot]=old_sign
        outcome=1 if int(rng.integers(2))==0 else -1
        self.x[pivot,:]=0;self.z[pivot,:]=0
        self.z[pivot,q]=1;self.sign[pivot]=(outcome==-1)
        return outcome,0.5

    def entropy(self, subsystem):
        """Von Neumann entropy in bits, for a pure stabilizer state.

        S(A) = |A| - dim(S_A) = rank(G restricted to complement B) - |B|.
        This computes PHYSICAL entanglement, not a CAMPS residual MPS bond.
        """
        a = np.array(sorted(set(subsystem)), dtype=int)
        b = np.array([i for i in range(self.n) if i not in set(a)], dtype=int)
        if not len(a) or not len(b): return 0
        matrix = np.concatenate((self.x[:self.n, b], self.z[:self.n, b]), axis=1).copy()
        rank = 0
        for col in range(matrix.shape[1]):
            piv = np.flatnonzero(matrix[rank:, col])
            if not len(piv): continue
            j = rank + int(piv[0])
            if j != rank: matrix[[rank,j]] = matrix[[j,rank]]
            for k in range(rank+1, self.n):
                if matrix[k,col]: matrix[k] ^= matrix[rank]
            rank += 1
            if rank == self.n: break
        result = rank-len(b)
        if result < 0 or result > min(len(a),len(b)):
            raise RuntimeError('Invalid entropy; tableau is inconsistent')
        return int(result)


def run_trajectory(n, depth, p, rng, words=None, record=False):
    """Simulate depth FULL brickwork cycles, returning entanglement and metadata.

    Every full cycle: odd bonds, measurement attempt on each site, even bonds,
    measurement attempt on each site. Independent Haar-equivalent 2-design
    samples are taken for each two-qubit gate. All measurements are conditioned.
    """
    words = two_qubit_clifford_words() if words is None else words
    tab = FastStabilizer(n)
    half = list(range(n//2))
    history = [tab.entropy(half)] if record else None
    measurements = 0; random_outcomes = 0
    for t in range(depth):
        for parity in (0,1):
            for a in range(parity,n-1,2):
                tab.random_two_qubit_clifford(a,a+1,rng,words)
            for q in range(n):
                if rng.random() < p:
                    _, probability = tab.measure_z(q,rng)
                    measurements += 1
                    random_outcomes += probability == 0.5
            if record: history.append(tab.entropy(half))
    return {'entropy':tab.entropy(half), 'history':history,
            'measurements':measurements,'random_outcomes':random_outcomes,
            'tableau':tab}
