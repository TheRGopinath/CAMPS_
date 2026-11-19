"""Checks for the QRC reservoir built on online CAMPS."""
import numpy as np

from qrc import Reservoir, feature_names, features, fit_ridge, predict, r2_scores


def test_swap_example_matches_hand_calculation():
    """U = SWAP: <m2> = cos t1, <m3> = cos t2, <m4> = cos t1 cos t3, <m2 m4> = cos t3."""
    res = Reservoir(unitary="swap", n_inputs=4, n_flush=1)
    th = [0.7, 1.9, 2.6, 0.4]
    c = np.cos(th)
    names, pairs = feature_names(res, "all")
    f = dict(zip(names, features(*res.outcome_distribution(th), pairs)))
    assert abs(f["<m1>"] - 1) < 1e-12
    assert abs(f["<m2>"] - c[0]) < 1e-12
    assert abs(f["<m3>"] - c[1]) < 1e-12
    assert abs(f["<m4>"] - c[0] * c[2]) < 1e-12
    assert abs(f["<m2 m4>"] - c[2]) < 1e-12
    assert abs(f["<m3 m5>"] - c[3]) < 1e-12


def test_camps_equals_dense_distribution():
    worst = 0.0
    for n_mem, n_data, t_prob, layout in ((1, 1, 0.0, "block"), (2, 1, 0.5, "block"), (1, 2, 0.5, "block"),
                                          (3, 1, 0.3, "block"), (2, 2, 0.0, "interleaved"),
                                          (3, 2, 0.3, "interleaved")):
        for seed in range(3):
            res = Reservoir(n_mem=n_mem, n_data=n_data, seed=seed, t_prob=t_prob, layout=layout)
            th = np.random.default_rng(seed).choice(res.angles, res.n_inputs)
            Mc, wc = res.outcome_distribution(th, "camps")
            Md, wd = res.outcome_distribution(th, "dense")
            dc = {tuple(m): p for m, p in zip(Mc, wc)}
            dd = {tuple(m): p for m, p in zip(Md, wd)}
            worst = max(worst, max(abs(dc.get(k, 0) - dd.get(k, 0)) for k in set(dc) | set(dd)))
    assert worst < 1e-10, worst


def test_literal_shots_match_exact_features():
    res = Reservoir(seed=3, t_prob=0.5)
    th = res.angles[[3, 17, 29]]
    _, pairs = feature_names(res, "all")
    exact = features(*res.outcome_distribution(th), pairs)
    t = 3000
    shots = res.run_shots(th, t, np.random.default_rng(1))
    est = features(shots, np.full(t, 1 / t), pairs)
    z = np.abs(est - exact) / np.sqrt(np.maximum(1 - exact ** 2, 1e-12) / t)
    assert z.max() < 4.5, z.max()


def test_swap_ridge_recovers_all_inputs_with_lag2():
    res = Reservoir(unitary="swap")
    rng = np.random.default_rng(0)
    th = res.angles[rng.integers(0, 32, size=(240, 3))]
    for kind, expect_last in (("lag1", False), ("lag2", True)):
        _, pairs = feature_names(res, kind)
        F = np.array([features(*res.outcome_distribution(t), pairs) for t in th])
        r2 = r2_scores(th[160:], predict(fit_ridge(F[:160], th[:160]), F[160:]))
        assert r2[0] > 0.95 and r2[1] > 0.95
        assert (r2[2] > 0.95) == expect_last, (kind, r2)


if __name__ == "__main__":
    for check in (test_swap_example_matches_hand_calculation, test_camps_equals_dense_distribution,
                  test_literal_shots_match_exact_features, test_swap_ridge_recovers_all_inputs_with_lag2):
        check()
        print(check.__name__, "passed")
