"""Evasive GPS-spoofing attacker (the 'attack' layer).

The attacker injects a spoofing offset d_k into the GPS measurement to redirect
the UAV toward an attacker-chosen destination while keeping the chi-square
detector below its threshold. Because the injected offset enters the innovation
directly,

    nu_k(d_k) = nu_k^0 + d_k        (nu_k^0 = innovation with no attack)

it shifts both the filter's posterior estimate (steering the UAV, since the MPC
controls on the estimate) and the detector statistic (risking detection).

Per-step constrained optimization solved by the attacker:

    minimize_{d}   || b0 + (C K) d  -  b_target ||^2
    subject to     (nu0 + d)^T S^{-1} (nu0 + d) <= tau      (chi-square stealth)
                   ||d|| <= d_max                            (magnitude cap)
                   ||d - d_prev|| <= delta_max               (ramp-rate cap)

Objective rationale (induced-bias / hijack target):
    The closed-loop MPC drives the *estimate* to the legitimate goal g. The true
    position satisfies x_true ~= x_hat - b, where b = (estimate - true) is the
    bias the attack induces. So at convergence x_true ~= g - b. To park the true
    UAV at the attacker goal g_att, the attacker drives the induced position bias
    b toward
        b_target = g_legit - g_att.
    Here b0 = C x_pred + C K nu0 - C x_true is the no-attack bias and (C K) d is
    the attacker's per-step leverage on it.

Constraint set:
    * stealth: a convex quadratic (ellipsoid) inequality in d, since S^{-1} > 0.
    * magnitude / ramp: convex norm-ball inequalities.
    The full program is a convex QCQP; we solve it with scipy SLSQP, which
    handles the nonlinear (quadratic) stealth constraint directly. Analytic
    gradients are supplied for speed and robustness. On solver failure the
    attacker falls back to d = 0 (inject nothing this step) -- the safe stealthy
    choice.

This implements the STRONG / omniscient attacker: it observes the clean
innovation nu0 (equivalently, the realized clean measurement and the true
position). This is the standard worst-case-for-defender baseline; weaker,
partial-knowledge attackers can only do worse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class AttackerConfig:
    """Explicit attacker interface (every knob the spec calls for)."""

    goal_pos: np.ndarray      # attacker destination g_att (position, R^2)
    legit_goal_pos: np.ndarray  # legitimate goal g (position, R^2)
    attack_start: int         # first step at which spoofing is injected
    d_max: float              # maximum spoofing magnitude ||d_k||
    delta_max: float          # maximum spoofing change per step ||d_k - d_{k-1}||
    tau: float                # chi-square stealth threshold (from the detector)

    def b_target(self) -> np.ndarray:
        """Induced-bias target b* = g_legit - g_att (position space)."""
        return np.asarray(self.legit_goal_pos, float) - np.asarray(self.goal_pos, float)


class EvasiveAttacker:
    """Solves the per-step stealthy spoofing optimization with SLSQP."""

    def __init__(self, C: np.ndarray, config: AttackerConfig) -> None:
        self.C = C
        self.cfg = config
        self._b_target = config.b_target()

    def compute(
        self,
        k: int,
        x_pred: np.ndarray,
        S: np.ndarray,
        K: np.ndarray,
        nu0: np.ndarray,
        x_true: np.ndarray,
        d_prev: np.ndarray,
    ) -> np.ndarray:
        """Return the spoofing offset d_k for this step (zeros before start).

        Args:
            k: Current step index.
            x_pred: Filter prior mean x_hat_{k|k-1}.
            S: Innovation covariance.
            K: Kalman gain.
            nu0: Clean (no-attack) innovation nu_k^0.
            x_true: True state (omniscient attacker observes position via C).
            d_prev: Offset injected at the previous step.
        """
        if k < self.cfg.attack_start:
            return np.zeros(self.C.shape[0])

        cfg = self.cfg
        M = self.C @ K                                  # (2x2) leverage on bias
        # No-attack induced bias b0 = C x_pred + C K nu0 - C x_true.
        b0 = self.C @ x_pred + M @ nu0 - self.C @ x_true
        Sinv = np.linalg.inv(S)
        bt = self._b_target

        def obj(d):
            r = b0 + M @ d - bt
            return float(r @ r)

        def obj_jac(d):
            r = b0 + M @ d - bt
            return 2.0 * (M.T @ r)

        # Inequality constraints g(d) >= 0.
        def c_stealth(d):
            e = nu0 + d
            return cfg.tau - float(e @ Sinv @ e)

        def c_stealth_jac(d):
            e = nu0 + d
            return -2.0 * (Sinv @ e)

        def c_mag(d):
            return cfg.d_max**2 - float(d @ d)

        def c_mag_jac(d):
            return -2.0 * d

        def c_ramp(d):
            diff = d - d_prev
            return cfg.delta_max**2 - float(diff @ diff)

        def c_ramp_jac(d):
            return -2.0 * (d - d_prev)

        constraints = [
            {"type": "ineq", "fun": c_stealth, "jac": c_stealth_jac},
            {"type": "ineq", "fun": c_mag, "jac": c_mag_jac},
            {"type": "ineq", "fun": c_ramp, "jac": c_ramp_jac},
        ]

        # Warm start from the previous offset (respects the ramp neighborhood).
        res = minimize(obj, x0=d_prev, jac=obj_jac, constraints=constraints,
                       method="SLSQP", options={"maxiter": 100, "ftol": 1e-9})

        if not res.success or not np.all(np.isfinite(res.x)):
            return np.zeros(self.C.shape[0])  # safe fallback: inject nothing
        return res.x
