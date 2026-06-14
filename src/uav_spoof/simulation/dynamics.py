"""Discrete-time linear UAV dynamics (the 'simulation' / true-system layer).

We model the UAV as a 2D double integrator (constant-acceleration kinematics):

    state   x = [px, py, vx, vy]^T   in R^4
    control u = [ax, ay]^T           in R^2   (commanded acceleration)
    output  y = [px, py]^T           in R^2   (GPS position measurement)

Discrete-time evolution with step dt:

    x_{k+1} = A x_k + B u_k + w_k        w_k ~ N(0, Sigma_w)
    y_k     = C x_k + d_k + v_k          v_k ~ N(0, Sigma_v)

d_k is the attack vector injected into the *measurement* (zero when no attacker
is present). GPS corrupts the position channel, which is exactly what C reads.

This module is the only place that samples real process/measurement noise and
advances the true state. Estimation, control, detection, and attack layers never
touch the true state directly -- they only see measurements, exactly as a real
onboard stack would.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LinearUAV:
    """Discrete-time linear UAV model with Gaussian process/measurement noise.

    Attributes:
        dt: Sampling period (s).
        sigma_w_pos: Process-noise std on each position state.
        sigma_w_vel: Process-noise std on each velocity state.
        sigma_v: Measurement-noise std on each GPS position channel (m).
        rng: NumPy random generator used for all noise sampling (seed it for
            reproducibility).
    """

    dt: float = 0.1
    sigma_w_pos: float = 0.02
    sigma_w_vel: float = 0.05
    sigma_v: float = 0.5
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    A: np.ndarray = field(init=False)
    B: np.ndarray = field(init=False)
    C: np.ndarray = field(init=False)
    Sigma_w: np.ndarray = field(init=False)
    Sigma_v: np.ndarray = field(init=False)

    n: int = field(init=False)  # state dimension
    l: int = field(init=False)  # control dimension
    m: int = field(init=False)  # measurement dimension

    def __post_init__(self) -> None:
        dt = self.dt
        I2 = np.eye(2)
        Z2 = np.zeros((2, 2))

        # Double-integrator: position integrates velocity, velocity integrates
        # commanded acceleration.
        self.A = np.block([[I2, dt * I2],
                           [Z2, I2]])
        self.B = np.block([[0.5 * dt**2 * I2],
                           [dt * I2]])
        self.C = np.block([I2, Z2])  # GPS observes position only.

        # Diagonal covariances. The position process-noise floor keeps Sigma_w
        # strictly positive definite, which the filter relies on for stable
        # covariance inverses.
        self.Sigma_w = np.diag([self.sigma_w_pos**2, self.sigma_w_pos**2,
                                self.sigma_w_vel**2, self.sigma_w_vel**2])
        self.Sigma_v = np.diag([self.sigma_v**2, self.sigma_v**2])

        self.n, self.l = self.B.shape
        self.m = self.C.shape[0]

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Propagate the TRUE state one step with freshly sampled process noise."""
        w = self.rng.multivariate_normal(np.zeros(self.n), self.Sigma_w)
        return self.A @ x + self.B @ u + w

    def measure(self, x: np.ndarray, d: np.ndarray | None = None) -> np.ndarray:
        """Produce a GPS measurement of the true state, optionally attacked.

        Args:
            x: True state.
            d: Optional attack vector added to the measurement (spoofing).

        Returns:
            Noisy (and possibly spoofed) position measurement.
        """
        v = self.rng.multivariate_normal(np.zeros(self.m), self.Sigma_v)
        y = self.C @ x + v
        if d is not None:
            y = y + d
        return y
