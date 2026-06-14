"""Tests for the Kalman filter."""

import numpy as np

from uav_spoof.estimation.kalman import KalmanFilter
from uav_spoof.simulation.dynamics import LinearUAV


def test_noiseless_perfect_init_tracks_exactly():
    # Noiseless TRUE system + exact init => the innovation is identically zero,
    # so the estimate stays exact regardless of gain. The filter is given a tiny
    # positive assumed-noise model so its innovation covariance is invertible
    # (a realistic configuration: assumed noise need not equal true noise).
    true_sys = LinearUAV(dt=0.1, sigma_w_pos=0.0, sigma_w_vel=0.0, sigma_v=0.0)
    model = LinearUAV(dt=0.1, sigma_w_pos=1e-3, sigma_w_vel=1e-3, sigma_v=1e-3)
    x = np.array([0.0, 0.0, 1.0, 1.0])
    kf = KalmanFilter(model, x0=x.copy(), P0=1e-6 * np.eye(4))
    u_prev = np.zeros(2)
    for _ in range(20):
        x = true_sys.step(x, u_prev)
        y = true_sys.measure(x)
        s = kf.step(u_prev, y)
        np.testing.assert_allclose(s.x_upd, x, atol=1e-6)


def test_innovation_covariance_symmetric_pd():
    uav = LinearUAV(dt=0.1, rng=np.random.default_rng(1))
    kf = KalmanFilter(uav, x0=np.zeros(4), P0=np.eye(4))
    x = np.zeros(4)
    s = kf.step(np.zeros(2), uav.measure(x))
    np.testing.assert_allclose(s.S, s.S.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(s.S) > 0)


def test_filter_beats_raw_measurements_on_average():
    # Statistical: filtered RMSE should be below raw-GPS RMSE on a seeded run.
    uav = LinearUAV(dt=0.1, rng=np.random.default_rng(123))
    x = np.array([0.0, 0.0, 1.0, 0.5])
    kf = KalmanFilter(uav, x0=np.array([1.0, 1.0, 0.0, 0.0]),
                      P0=np.diag([2.0, 2.0, 1.0, 1.0]))
    u = np.zeros(2)
    ef, em = [], []
    for k in range(300):
        x = uav.step(x, u)
        y = uav.measure(x)
        s = kf.step(u, y)
        if k > 50:
            ef.append(np.linalg.norm(s.x_upd[:2] - x[:2]))
            em.append(np.linalg.norm(y - x[:2]))
    assert np.sqrt(np.mean(np.square(ef))) < np.sqrt(np.mean(np.square(em)))
