"""Shared simulation runner for the Stage 1 (Kalman filter) experiments.

All three Stage 1 experiments ask questions about the SAME underlying scenario:
an open-loop, maneuvering UAV under no attack, observed through noisy GPS and
tracked by the Kalman filter. Centralizing the run here keeps the three
experiment scripts independently runnable while guaranteeing they describe the
same system. No attacker is present anywhere in Stage 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_spoof.estimation.kalman import KalmanFilter
from uav_spoof.simulation.dynamics import LinearUAV


@dataclass
class TrackingRun:
    t: np.ndarray            # time vector (s)
    true: np.ndarray         # (T, 4) true states
    est: np.ndarray          # (T, 4) KF posterior estimates
    meas: np.ndarray         # (T, 2) raw GPS measurements
    nis: np.ndarray          # (T,) normalized innovation squared
    P_upd: np.ndarray        # (T, 4, 4) posterior covariances
    burn_in: int             # steps to discard for steady-state metrics


def run_tracking(seed: int, steps: int = 400, burn_in: int = 50) -> TrackingRun:
    """Simulate open-loop maneuvering flight tracked by the Kalman filter.

    The control is a fixed sinusoidal acceleration profile (known to the filter),
    so the velocity is genuinely time-varying -- a non-trivial tracking task.
    The filter is initialized deliberately away from the true state to exercise
    convergence rather than assume it.
    """
    rng = np.random.default_rng(seed)
    uav = LinearUAV(dt=0.1, rng=rng)

    x_true = np.array([0.0, 0.0, 1.0, 0.5])
    t = np.arange(steps) * uav.dt
    # Open-loop maneuvering acceleration in both axes.
    U = np.stack([0.6 * np.sin(0.5 * t), 0.4 * np.cos(0.3 * t)], axis=1)

    # Wrong initial guess + loose covariance so convergence is actually tested.
    x0 = np.array([3.0, -2.0, 0.0, 0.0])
    P0 = np.diag([5.0, 5.0, 2.0, 2.0])
    kf = KalmanFilter(uav, x0, P0)

    true_hist = np.empty((steps, 4))
    est_hist = np.empty((steps, 4))
    meas_hist = np.empty((steps, 2))
    nis_hist = np.empty(steps)
    P_hist = np.empty((steps, 4, 4))

    u_prev = np.zeros(uav.l)
    for k in range(steps):
        x_true = uav.step(x_true, U[k])
        y = uav.measure(x_true)              # no attack
        s = kf.step(u_prev, y)
        u_prev = U[k]

        true_hist[k] = x_true
        est_hist[k] = s.x_upd
        meas_hist[k] = y
        nis_hist[k] = float(s.innovation @ np.linalg.solve(s.S, s.innovation))
        P_hist[k] = s.P_upd

    return TrackingRun(t, true_hist, est_hist, meas_hist, nis_hist, P_hist, burn_in)
