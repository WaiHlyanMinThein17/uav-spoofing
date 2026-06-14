"""Tests for the UAV dynamics model."""

import numpy as np

from uav_spoof.simulation.dynamics import LinearUAV


def test_matrix_shapes():
    uav = LinearUAV(dt=0.1)
    assert uav.A.shape == (4, 4)
    assert uav.B.shape == (4, 2)
    assert uav.C.shape == (2, 4)
    assert (uav.n, uav.l, uav.m) == (4, 2, 2)


def test_noiseless_propagation_matches_linear_model():
    # With zero noise, step() must equal the deterministic A x + B u.
    uav = LinearUAV(dt=0.1, sigma_w_pos=0.0, sigma_w_vel=0.0, sigma_v=0.0)
    x = np.array([1.0, -2.0, 0.5, 0.3])
    u = np.array([0.2, -0.1])
    expected = uav.A @ x + uav.B @ u
    np.testing.assert_allclose(uav.step(x, u), expected, atol=1e-12)


def test_measurement_observes_position_only():
    uav = LinearUAV(dt=0.1, sigma_v=0.0)
    x = np.array([3.0, 4.0, 9.0, 9.0])
    y = uav.measure(x)
    np.testing.assert_allclose(y, x[:2], atol=1e-12)


def test_attack_vector_is_added_to_measurement():
    uav = LinearUAV(dt=0.1, sigma_v=0.0)
    x = np.array([3.0, 4.0, 0.0, 0.0])
    d = np.array([1.5, -2.0])
    np.testing.assert_allclose(uav.measure(x, d=d), x[:2] + d, atol=1e-12)
