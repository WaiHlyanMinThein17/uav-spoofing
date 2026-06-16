"""Tests for the chi-square spoofing detector."""

import numpy as np
from scipy.stats import chi2

from uav_spoof.detection.chi_square import ChiSquareDetector


def test_threshold_is_chi2_quantile():
    det = ChiSquareDetector(dof=2, alpha=0.05)
    assert np.isclose(det.threshold, chi2.ppf(0.95, 2))


def test_statistic_matches_quadratic_form():
    det = ChiSquareDetector(dof=2, alpha=0.05)
    nu = np.array([1.0, -2.0])
    S = np.array([[2.0, 0.3], [0.3, 1.5]])
    expected = float(nu @ np.linalg.solve(S, nu))
    assert np.isclose(det.statistic(nu, S), expected)


def test_alarm_fires_above_threshold_only():
    det = ChiSquareDetector(dof=2, alpha=0.05)
    S = np.eye(2)
    small = det.step(np.array([0.1, 0.1]), S)   # q = 0.02, well below ~5.99
    big = det.step(np.array([5.0, 5.0]), S)      # q = 50, well above
    assert small.alarm is False
    assert big.alarm is True


def test_false_positive_rate_matches_alpha():
    # Feed many whitened standard-normal innovations; the empirical alarm rate
    # should be close to alpha because q = nu^T nu ~ chi-square(2) exactly.
    rng = np.random.default_rng(0)
    alpha = 0.05
    det = ChiSquareDetector(dof=2, alpha=alpha)
    S = np.eye(2)
    alarms = [det.step(rng.standard_normal(2), S).alarm for _ in range(20000)]
    rate = np.mean(alarms)
    assert abs(rate - alpha) < 0.01


def test_invalid_alpha_rejected():
    for bad in (0.0, 1.0, -0.1, 2.0):
        try:
            ChiSquareDetector(dof=2, alpha=bad)
        except ValueError:
            continue
        raise AssertionError(f"alpha={bad} should have raised ValueError")
