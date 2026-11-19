"""Worker for examples/qrc_scan.py, kept in an importable module so the
parallel scan also works on Windows / in Jupyter (multiprocessing 'spawn'
can only send functions that live in a real module)."""
import numpy as np

from .reservoir import Reservoir, feature_names, features

_CACHE = {}


def parse(cfg):
    a, b = cfg.lower().split("x")
    return int(a), int(b)


def make_reservoir(cfg, r, p):
    """Reservoir r of config 'AxB' (A memory, B data qubits) for scan params p."""
    nm, nd = parse(cfg)
    return Reservoir(nm, nd, p["inputs"], p["flush"], p["lam"], "random", 1,
                     seed=int(np.random.SeedSequence([p["seed"], nm, nd, r]).generate_state(1)[0]),
                     t_prob=p["t_prob"], n_values=p["n_values"], layout=p["layout"])


def job(task):
    """One input sequence: t literal CAMPS shots -> (index, feature vector, CAMPS stats)."""
    cfg, r, i, thetas, p = task
    key = (cfg, r, tuple(sorted(p.items())))
    if key not in _CACHE:
        res = make_reservoir(cfg, r, p)
        _CACHE[key] = (res, feature_names(res, "all")[1])
    res, pairs = _CACHE[key]
    nm, nd = parse(cfg)
    rng = np.random.default_rng(np.random.SeedSequence([p["seed"], nm, nd, r, i, 7]))
    stats = {}
    records = res.run_shots(thetas, p["shots"], rng, stats)
    return i, features(records, np.full(p["shots"], 1.0 / p["shots"]), pairs), stats
