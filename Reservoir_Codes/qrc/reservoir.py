"""Quantum reservoir computing with a monitored random-Clifford reservoir.

Circuit (one shot, one input sequence theta_1..theta_K):

    qubits 0..n_mem-1          : memory register (never measured)
    qubits n_mem..n_mem+n_data-1 : data register

    for step k = 1 .. K + n_flush:
        RY(theta_k) on every data qubit        (theta_k = 0 on flush steps)
        U_k   = brickwork of random two-qubit Cliffords over all qubits
                (cycles_per_step full cycles; for 2 qubits: one Clifford)
                or SWAP(memory qubit 0, data qubit) in the SWAP example
        measure Z on every data qubit -> m_k = +/-1   (state collapses)

Every U_k is an independent uniformly random two-qubit Clifford (same sampler
as the MIPT code). The realisation {U_k} is drawn once per reservoir and then
kept fixed for all shots and all input sequences: the features must be a
deterministic function of the inputs for the trained readout to mean anything.

Flush steps (no input) at the end let information still held in the memory
reach a measurement, e.g. with SWAP the last input is only read one step later.

Inputs: theta = n pi / 2^lam, n = 0 .. n_values - 1 (default lam = 5, 32 values:
[0, pi)); lam = 6 gives [0, pi/2).

Optional T gates (t_prob > 0) make U non-Clifford; CAMPS absorbs them as
Z rotations by pi/4 (T = e^{i pi/8} RZ(pi/4)).
Features: <m_k> and two-time correlators <m_k m_l> over shots.
Readout: theta_hat = W f + b, trained by ridge regression.
"""
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from camps.online import OnlineCAMPS, clifford_block
from camps.reference import dense_gate
from mipt import two_qubit_clifford_words
from stabilizers_pbc import Gate

_WORD_GATES = {0: ("H", 0), 1: ("H", 1), 2: ("S", 0), 3: ("S", 1), 4: ("CX", 0, 1), 5: ("CX", 1, 0)}


def word_to_gates(word, a, b):
    q = (a, b)
    out = []
    for g in word:
        spec = _WORD_GATES[g]
        out.append(Gate(spec[0], q[spec[1]]) if len(spec) == 2 else Gate(spec[0], q[spec[1]], q[spec[2]]))
    return out


def swap_gates(a, b):
    return [Gate("CX", a, b), Gate("CX", b, a), Gate("CX", a, b)]


@dataclass
class Reservoir:
    n_mem: int = 1
    n_data: int = 1
    n_inputs: int = 3
    n_flush: int = 1
    lam: int = 5
    unitary: str = "random"          # "random" or "swap"
    cycles_per_step: int = 1
    seed: int = 0
    t_prob: float = 0.0              # chance of a T gate on each qubit after each Clifford
    n_values: int = 32               # theta = n pi / 2^lam, n = 0 .. n_values-1
    layout: str = "block"            # "block": memory then data; "interleaved": spread data along the chain
    unitaries: list = field(default=None, repr=False)

    def __post_init__(self):
        self.n = self.n_mem + self.n_data
        if self.n < 2:
            raise ValueError("need at least one memory and one data qubit")
        if self.layout == "block":
            self.data = list(range(self.n_mem, self.n))
        elif self.layout == "interleaved":
            # data qubits evenly spaced along the chain, so each has memory neighbours
            self.data = sorted({int(round((i + 0.5) * self.n / self.n_data - 0.5)) for i in range(self.n_data)})
        else:
            raise ValueError("layout must be 'block' or 'interleaved'")
        self.memory = [q for q in range(self.n) if q not in self.data]
        self.steps = self.n_inputs + self.n_flush
        if self.unitaries is None:
            self.unitaries = self._draw_unitaries()
        # Clifford stretches of each U_k precomputed once (T gates split them)
        self.segments = []
        for gates in self.unitaries:
            segs, run = [], []
            for g in gates:
                if g.kind == "T":
                    if run:
                        segs.append(("block", clifford_block(run, self.n), run)); run = []
                    segs.append(("T", g.a, None))
                else:
                    run.append(g)
            if run:
                segs.append(("block", clifford_block(run, self.n), run))
            self.segments.append(segs)

    # ------------------------------------------------------------------ circuit
    def _draw_unitaries(self):
        rng = np.random.default_rng(self.seed)
        if self.unitary == "swap":
            if self.n_mem != 1 or self.n_data != 1:
                raise ValueError("SWAP example uses one memory and one data qubit")
            return [swap_gates(self.memory[0], self.data[0]) for _ in range(self.steps)]
        words = two_qubit_clifford_words()
        out = []
        for _ in range(self.steps):
            gates = []
            for _ in range(self.cycles_per_step):
                for parity in (0, 1):
                    for a in range(parity, self.n - 1, 2):
                        gates += word_to_gates(words[int(rng.integers(len(words)))], a, a + 1)
                        for q in (a, a + 1):
                            if self.t_prob > 0 and rng.random() < self.t_prob:
                                gates.append(Gate("T", q))   # non-Clifford: handled by CAMPS
            out.append(gates)
        return out

    @property
    def angles(self):
        return np.arange(self.n_values) * np.pi / 2 ** self.lam

    def step_angles(self, thetas):
        thetas = list(thetas)
        if len(thetas) != self.n_inputs:
            raise ValueError(f"expected {self.n_inputs} input angles")
        return thetas + [0.0] * self.n_flush

    @property
    def n_outcomes(self):
        return self.steps * self.n_data          # column = step * n_data + data index

    # ------------------------------------------------------------------ exact outcome distribution
    def outcome_distribution(self, thetas, backend="camps", stats=None):
        """All measurement records and their exact probabilities, by branching
        on every measurement. Returns (M, w): M[s] = record of +/-1, w[s] = prob."""
        if backend == "camps":
            branches = [(OnlineCAMPS(self.n), (), 1.0)]
        elif backend == "dense":
            v = np.zeros(2 ** self.n, complex); v[0] = 1
            branches = [(v, (), 1.0)]
        else:
            raise ValueError(backend)
        for k, theta in enumerate(self.step_angles(thetas)):
            new = []
            for sim, rec, prob in branches:
                sim = _apply_step(sim, self, k, theta, backend)
                parts = [(sim, rec, prob)]
                for q in self.data:
                    split = []
                    for s, r, pr in parts:
                        for m in (1, -1):
                            s2, pm = _measure(s, q, m, self.n, backend)
                            if pm > 1e-14:
                                split.append((s2, r + (m,), pr * pm))
                    parts = split
                new += parts
            branches = new
        if stats is not None and backend == "camps":
            for sim, _, _ in branches:
                for key, c in sim.counts.items():
                    stats[key] = stats.get(key, 0) + c
                stats["max_bond"] = max(stats.get("max_bond", 1), sim.max_bond_seen)
        M = np.array([r for _, r, _ in branches], dtype=np.int8)
        w = np.array([p for _, _, p in branches])
        return M, w / w.sum()

    # ------------------------------------------------------------------ literal shots
    def run_shots(self, thetas, t, rng, stats=None):
        """Run t independent CAMPS trajectories (sampling every measurement).
        Statistically identical to sampling t records from outcome_distribution."""
        out = np.empty((t, self.n_outcomes), dtype=np.int8)
        for shot in range(t):
            sim = OnlineCAMPS(self.n)
            col = 0
            for k, theta in enumerate(self.step_angles(thetas)):
                sim = _apply_step(sim, self, k, theta, "camps")
                for q in self.data:
                    out[shot, col], _ = sim.measure_z(q, rng)
                    col += 1
            if stats is not None:
                for key, c in sim.counts.items():
                    stats[key] = stats.get(key, 0) + c
                stats["max_bond"] = max(stats.get("max_bond", 1), sim.max_bond_seen)
        return out


def _apply_step(sim, res, k, theta, backend):
    if backend == "camps":
        sim = sim.copy()
        if theta != 0.0:
            for q in res.data:
                sim.rotation(q, "Y", theta)
        for kind, payload, gates in res.segments[k]:
            if kind == "T":
                sim.rotation(payload, "Z", np.pi / 4)
            else:
                sim.clifford_block(payload, gates)
        return sim
    v = sim
    if theta != 0.0:
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        ry = np.array([[c, -s], [s, c]])
        for q in res.data:
            t = v.reshape((2,) * res.n)
            v = np.moveaxis(np.tensordot(ry, t, axes=(1, q)), 0, q).reshape(-1)
    for g in res.unitaries[k]:
        v = dense_gate(v, g, res.n)
    return v


def _measure(sim, q, m, n, backend):
    if backend == "camps":
        if m == 1:
            p = sim.z_probability(q)
        else:
            p = 1 - sim.z_probability(q)
        if p <= 1e-14:
            return sim, 0.0
        s2 = sim.copy()
        _, p = s2.measure_z(q, outcome=m)
        return s2, p
    t = sim.reshape((2,) * n).copy()
    idx = [slice(None)] * n
    idx[q] = 1 if m == 1 else 0
    t[tuple(idx)] = 0
    w = t.reshape(-1)
    p = float(np.vdot(w, w).real)
    return (w / np.sqrt(p) if p > 0 else w), p


# ---------------------------------------------------------------------- features
def feature_names(res, kind="all"):
    """Names of the features: singles <m_k> plus selected two-time correlators.

    kind: 'singles' | 'lag1' (note's choice: consecutive steps) | 'lag2' |
          'all' (every pair; ridge regression decides which matter)."""
    cols = [(k, d) for k in range(res.steps) for d in range(res.n_data)]
    names = [f"<m{k+1}>" if res.n_data == 1 else f"<m{k+1},{d}>" for k, d in cols]
    pairs = []
    for (i, (ki, di)), (j, (kj, dj)) in combinations(enumerate(cols), 2):
        lag = kj - ki
        if kind == "all" or (kind == "lag1" and lag == 1 and di == dj) or (kind == "lag2" and lag == 2 and di == dj):
            pairs.append((i, j))
    if kind == "singles":
        pairs = []
    lab = lambda k, d: f"m{k+1}" if res.n_data == 1 else f"m{k+1},{d}"
    names += [f"<{lab(*cols[i])} {lab(*cols[j])}>" for i, j in pairs]
    return names, pairs


def features(M, w, pairs):
    """Expectation values from records M (+/-1) with weights w (sum 1)."""
    Mf = M.astype(float)
    singles = w @ Mf
    corr = np.array([w @ (Mf[:, i] * Mf[:, j]) for i, j in pairs])
    return np.concatenate([singles, corr])


def shot_weights(w, t, rng):
    """Empirical weights of t shots drawn from the exact distribution w."""
    return rng.multinomial(t, w) / t


# ---------------------------------------------------------------------- ridge regression
def fit_ridge(F, Y, lambdas=None, folds=5, rng=None):
    """Ridge regression Y ~ F W + b with lambda chosen by k-fold CV.

    Features are standardised with training statistics; constant features are
    dropped. Returns a dict model usable by predict()."""
    lambdas = np.logspace(-8, 3, 23) if lambdas is None else lambdas
    rng = np.random.default_rng(0) if rng is None else rng
    mu, sd = F.mean(0), F.std(0)
    keep = sd > 1e-12
    X = (F[:, keep] - mu[keep]) / sd[keep]
    ym = Y.mean(0)
    Yc = Y - ym

    def solve(Xa, Ya, lam):
        return np.linalg.solve(Xa.T @ Xa + lam * np.eye(Xa.shape[1]), Xa.T @ Ya)

    idx = rng.permutation(len(X))
    chunks = np.array_split(idx, folds)
    cv = []
    for lam in lambdas:
        err = 0.0
        for c in chunks:
            tr = np.setdiff1d(idx, c)
            W = solve(X[tr], Yc[tr], lam)
            err += np.sum((X[c] @ W - Yc[c]) ** 2)
        cv.append(err / Y.size)
    best = float(lambdas[int(np.argmin(cv))])
    W = solve(X, Yc, best)
    return {"mu": mu, "sd": sd, "keep": keep, "W": W, "b": ym, "lambda": best, "cv_mse": float(min(cv))}


def predict(model, F):
    X = (F[:, model["keep"]] - model["mu"][model["keep"]]) / model["sd"][model["keep"]]
    return X @ model["W"] + model["b"]


def r2_scores(Y, Yhat):
    ss_res = ((Y - Yhat) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    return 1 - ss_res / ss_tot
