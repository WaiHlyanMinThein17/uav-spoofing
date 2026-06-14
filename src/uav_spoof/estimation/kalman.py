"""Linear Kalman filter for UAV state estimation (the 'estimation' layer).

Two steps per timestep:

  Predict:  xhat_{k|k-1} = A xhat_{k-1} + B u_{k-1}
            P_{k|k-1}    = A P_{k-1} A^T + Sigma_w
  Update:   S_k  = C P_{k|k-1} C^T + R          (innovation covariance)
            K_k  = P_{k|k-1} C^T S_k^{-1}        (Kalman gain)
            nu_k = y_k - C xhat_{k|k-1}          (innovation / residual)
            xhat_{k|k} = xhat_{k|k-1} + K_k nu_k
            P_{k|k}    = (I - K_k C) P_{k|k-1}

Why the gain balances prediction against measurement: K_k = P_pred C^T S^{-1}.
If measurement noise R dominates S, K -> 0 and the update trusts the model;
if prediction covariance P_pred dominates, K grows and the update defers to the
sensor. The gain is the minimum-variance compromise between the two.

The innovation nu_k and its covariance S_k are exactly the ingredients the
chi-square spoofing detector consumes in stage 3, so we surface them in KFStep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_spoof.simulation.dynamics import LinearUAV


@dataclass
class KFStep:
    """Diagnostics captured at one filter step (for plotting / detection)."""

    x_pred: np.ndarray      # xhat_{k|k-1}, prior mean
    P_pred: np.ndarray      # P_{k|k-1}, prior covariance
    innovation: np.ndarray  # nu_k = y_k - C xhat_{k|k-1}
    S: np.ndarray           # innovation covariance C P_{k|k-1} C^T + R
    x_upd: np.ndarray       # xhat_{k|k}, posterior mean
    P_upd: np.ndarray       # P_{k|k}, posterior covariance


class KalmanFilter:
    """Standard linear Kalman filter parameterized by a LinearUAV model."""

    def __init__(self, uav: LinearUAV, x0: np.ndarray, P0: np.ndarray) -> None:
        """Initialize the filter.

        Args:
            uav: Model supplying A, B, C and the noise covariances the filter
                assumes (here, the true ones -- a matched filter).
            x0: Initial posterior mean estimate.
            P0: Initial posterior covariance (encodes initial uncertainty).
        """
        self.A, self.B, self.C = uav.A, uav.B, uav.C
        self.Q = uav.Sigma_w          # process-noise covariance assumed by filter
        self.R = uav.Sigma_v          # measurement-noise covariance assumed by filter
        self.x = np.asarray(x0, dtype=float)  # xhat_{k|k}
        self.P = np.asarray(P0, dtype=float)  # P_{k|k}
        self._I = np.eye(self.A.shape[0])

    def predict(self, u_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute the prior (predicted) mean and covariance WITHOUT mutating.

        Exposed so an external agent (e.g. the Stage-4 attacker) can reason about
        the filter's prior and gain before a measurement is processed, without
        duplicating the prediction math or touching filter state.
        """
        x_pred = self.A @ self.x + self.B @ u_prev
        P_pred = self.A @ self.P @ self.A.T + self.Q
        return x_pred, P_pred

    def gain(self, P_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (S, K): innovation covariance and Kalman gain for a given prior.

        S = C P_pred C^T + R,   K = P_pred C^T S^{-1}.
        """
        S = self.C @ P_pred @ self.C.T + self.R
        K = P_pred @ self.C.T @ np.linalg.inv(S)
        return S, K

    def step(self, u_prev: np.ndarray, y: np.ndarray) -> KFStep:
        """Advance one timestep given the previous control and new measurement.

        Args:
            u_prev: Control applied over the interval ending at this step.
            y: Incoming (possibly spoofed) GPS measurement.

        Returns:
            KFStep with prior, innovation, and posterior diagnostics.
        """
        # --- predict ---
        x_pred, P_pred = self.predict(u_prev)

        # --- update ---
        S, K = self.gain(P_pred)
        innovation = y - self.C @ x_pred
        x_upd = x_pred + K @ innovation
        P_upd = (self._I - K @ self.C) @ P_pred
        P_upd = 0.5 * (P_upd + P_upd.T)  # symmetrize against round-off drift

        self.x, self.P = x_upd, P_upd
        return KFStep(x_pred, P_pred, innovation, S, x_upd, P_upd)
