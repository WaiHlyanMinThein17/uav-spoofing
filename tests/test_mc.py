"""Tests for the Monte Carlo evaluation harness."""

import numpy as np

from experiments.mc import (aggregate, make_seeds, run_monte_carlo)


def test_make_seeds_reproducible_and_independent():
    s1 = make_seeds(12345, 50)
    s2 = make_seeds(12345, 50)
    assert s1 == s2                      # reproducible from the same entropy
    assert len(set(s1)) == 50            # independent (no collisions)
    assert make_seeds(999, 50) != s1     # different entropy -> different seeds


def test_aggregate_continuous_matches_numpy():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    a = aggregate(vals, proportion=False)
    assert np.isclose(a.mean, 3.0)
    assert np.isclose(a.std, np.std(vals, ddof=1))
    assert a.min == 1.0 and a.max == 5.0
    assert a.ci_method == "student_t"
    assert a.ci95_low < a.mean < a.ci95_high


def test_aggregate_proportion_uses_wilson_within_unit_interval():
    vals = [1.0] * 80 + [0.0] * 20       # 80% successes
    a = aggregate(vals, proportion=True)
    assert np.isclose(a.mean, 0.8)
    assert a.ci_method == "wilson"
    assert 0.0 <= a.ci95_low <= a.mean <= a.ci95_high <= 1.0


def test_run_monte_carlo_collects_all_trials_and_keys():
    # Trial metric is deterministic in the seed so we can check aggregation.
    def trial(seed: int) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        return {"x": float(rng.normal()), "flag": float(seed % 2)}

    res = run_monte_carlo(trial, base_entropy=7, n_trials=30,
                          proportion_keys=["flag"])
    assert len(res.per_trial) == 30
    assert len(res.seeds) == 30
    assert set(res.aggregates) == {"x", "flag"}
    assert res.aggregates["flag"].ci_method == "wilson"
    assert res.aggregates["x"].ci_method == "student_t"
