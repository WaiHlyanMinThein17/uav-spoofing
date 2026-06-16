"""Tests for the evasive attacker's per-step optimization."""

import numpy as np

from uav_spoof.attack.evasive import AttackerConfig, EvasiveAttacker


def _make(d_max=2.0, delta_max=1.0, tau=5.991, start=0):
    C = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    cfg = AttackerConfig(goal_pos=np.array([6.0, 6.0]),
                         legit_goal_pos=np.array([10.0, 10.0]),
                         attack_start=start, d_max=d_max, delta_max=delta_max, tau=tau)
    return C, EvasiveAttacker(C, cfg)


def _ctx():
    # Simple, well-conditioned prior/gain/innovation for testing.
    x_pred = np.array([1.0, 1.0, 0.5, 0.5])
    S = np.eye(2) * 0.5
    K = np.zeros((4, 2)); K[0, 0] = K[1, 1] = 0.6
    nu0 = np.array([0.1, -0.1])
    x_true = np.array([1.0, 1.0, 0.5, 0.5])
    return x_pred, S, K, nu0, x_true


def test_no_injection_before_attack_start():
    C, atk = _make(start=50)
    x_pred, S, K, nu0, x_true = _ctx()
    d = atk.compute(10, x_pred, S, K, nu0, x_true, d_prev=np.zeros(2))
    assert np.allclose(d, 0.0)


def test_respects_magnitude_and_stealth_constraints():
    C, atk = _make(d_max=1.5, delta_max=1.5, tau=5.991)
    x_pred, S, K, nu0, x_true = _ctx()
    d = atk.compute(0, x_pred, S, K, nu0, x_true, d_prev=np.zeros(2))
    # magnitude
    assert np.linalg.norm(d) <= 1.5 + 1e-6
    # stealth: q(d) = (nu0+d)^T S^-1 (nu0+d) <= tau
    e = nu0 + d
    q = float(e @ np.linalg.inv(S) @ e)
    assert q <= 5.991 + 1e-6


def test_respects_ramp_constraint():
    C, atk = _make(d_max=5.0, delta_max=0.3, tau=1e9)  # stealth slack; ramp binds
    x_pred, S, K, nu0, x_true = _ctx()
    d_prev = np.array([0.2, 0.0])
    d = atk.compute(0, x_pred, S, K, nu0, x_true, d_prev=d_prev)
    assert np.linalg.norm(d - d_prev) <= 0.3 + 1e-6


def test_moves_estimate_bias_toward_target_when_unconstrained():
    # With generous limits, the attack should reduce the gap between the induced
    # bias and the target bias b* = g_legit - g_att relative to no injection.
    C, atk = _make(d_max=10.0, delta_max=10.0, tau=1e9)
    x_pred, S, K, nu0, x_true = _ctx()
    M = C @ K
    b0 = C @ x_pred + M @ nu0 - C @ x_true
    b_target = np.array([10.0, 10.0]) - np.array([6.0, 6.0])
    gap0 = np.linalg.norm(b0 - b_target)
    d = atk.compute(0, x_pred, S, K, nu0, x_true, d_prev=np.zeros(2))
    gap = np.linalg.norm(b0 + M @ d - b_target)
    assert gap < gap0
