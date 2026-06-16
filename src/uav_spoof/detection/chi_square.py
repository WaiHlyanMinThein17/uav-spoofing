"""Chi-square GPS-spoofing detector (the 'detection' layer).

The detector watches the Kalman filter's innovation. Under the null hypothesis
"no attack" (d_k = 0), the innovation is zero-mean Gaussian, nu_k ~ N(0, S_k),
so the normalized innovation squared (NIS)

    q_k = nu_k^T S_k^{-1} nu_k

is chi-square distributed with m degrees of freedom (m = measurement dimension).
This is the formal basis of the test: whitening the innovation by S_k^{-1} turns
its energy into a sum of m squared standard normals.

Given a significance level alpha, the threshold is the (1 - alpha) quantile of
chi-square(m):

    tau = chi2.ppf(1 - alpha, m)

An alarm is raised when q_k > tau. Under the null, P(q_k > tau) = alpha, so the
false-positive rate equals alpha by construction -- provided the filter is
consistent, which Stage 1 Experiment 3 validates empirically. Under a spoofing
attack the innovation acquires a nonzero mean (the statistic becomes noncentral
chi-square), inflating q_k and triggering the alarm.

The detector is intentionally memoryless / per-sample: it consumes one
(innovation, S) pair and returns one decision. This keeps it a faithful
implementation of the chi-square hypothesis test and matches the NIS quantity
validated in Stage 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2


@dataclass
class DetectorResult:
    """Outcome of one detector evaluation."""

    statistic: float   # q_k = nu^T S^{-1} nu (the NIS value)
    alarm: bool        # True if q_k exceeds the threshold


class ChiSquareDetector:
    """Per-sample chi-square test on the Kalman innovation."""

    def __init__(self, dof: int, alpha: float = 0.05) -> None:
        """Initialize the detector.

        Args:
            dof: Degrees of freedom = measurement dimension m.
            alpha: Significance level (target false-positive rate). The detection
                threshold is the (1 - alpha) quantile of chi-square(dof).
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.dof = int(dof)
        self.alpha = float(alpha)
        self.threshold = float(chi2.ppf(1.0 - self.alpha, self.dof))

    def statistic(self, innovation: np.ndarray, S: np.ndarray) -> float:
        """Compute the NIS statistic q = nu^T S^{-1} nu.

        Solving S x = nu (rather than forming S^{-1}) is the numerically stable
        way to evaluate the quadratic form.
        """
        nu = np.asarray(innovation, dtype=float)
        return float(nu @ np.linalg.solve(np.asarray(S, dtype=float), nu))

    def step(self, innovation: np.ndarray, S: np.ndarray) -> DetectorResult:
        """Evaluate the detector on one innovation / covariance pair."""
        q = self.statistic(innovation, S)
        return DetectorResult(statistic=q, alarm=bool(q > self.threshold))
