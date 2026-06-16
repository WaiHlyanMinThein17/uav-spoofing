"""Shared runner for the Stage 3 (chi-square detector) experiments.

Reuses the Stage 1 open-loop maneuvering scenario (same matched Kalman filter),
adds a per-sample chi-square detector on the innovation, and optionally injects a
GPS-spoofing attack into the measurement. Two experiments consume this runner:

  * false-positive rate  -- no attack; the alarm rate should equal alpha.
  * detection rate       -- a naive large constant-offset attacker; the alarm
                            rate while the attack is active should be high.

The "naive" attacker here is the simplest possible spoofer: it adds a fixed
position offset to the GPS measurement from a chosen step onward. It makes no
attempt to stay stealthy -- that is the Stage 4 evasive attacker's job. The naive
attacker is the baseline against which detection rate is measured.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_spoof.detection.chi_square import ChiSquareDetector
from uav_spoof.estimation.kalman import KalmanFilter
from uav_spoof.simulation.dynamics import LinearUAV


@dataclass
class DetectionRun:
    t: np.ndarray            # (T,) time
    statistic: np.ndarray    # (T,) NIS statistic q_k
    alarm: np.ndarray        # (T,) bool alarms
    attacked: np.ndarray     # (T,) bool: was this step's measurement spoofed
    threshold: float         # detector threshold tau
    alpha: float             # significance level
    burn_in: int             # steps to discard before scoring
    attack_start: int        # first attacked step (T if no attack)


def naive_offset_attack(offset: np.ndarray, attack_start: int):
    """Build a naive attacker: constant measurement offset from attack_start on.

    Returns a function d(k) -> offset vector or None, suitable for passing to the
    simulation's measure(x, d=...) call.
    """
    offset = np.asarray(offset, dtype=float)

    def d(k: int):
        return offset if k >= attack_start else None

    return d


def run_detection(
    seed: int,
    steps: int = 400,
    burn_in: int = 50,
    alpha: float = 0.05,
    attack=None,
    attack_start: int = 10**9,
) -> DetectionRun:
    """Simulate maneuvering flight with the chi-square detector running.

    Args:
        seed: RNG seed for this trial.
        steps: Number of timesteps.
        burn_in: Steps to discard before scoring (filter convergence).
        alpha: Detector significance level.
        attack: Optional function k -> offset vector (or None) injected into the
            measurement. None means no attack at any step.
        attack_start: First attacked step, used only to label which steps are
            attacked in the output (the attack function controls the injection).
    """
    rng = np.random.default_rng(seed)
    uav = LinearUAV(dt=0.1, rng=rng)
    detector = ChiSquareDetector(dof=uav.m, alpha=alpha)

    x_true = np.array([0.0, 0.0, 1.0, 0.5])
    t = np.arange(steps) * uav.dt
    U = np.stack([0.6 * np.sin(0.5 * t), 0.4 * np.cos(0.3 * t)], axis=1)

    # Deliberately wrong initial guess so the filter must converge (as in Stage 1).
    kf = KalmanFilter(uav, x0=np.array([3.0, -2.0, 0.0, 0.0]),
                      P0=np.diag([5.0, 5.0, 2.0, 2.0]))

    stat = np.empty(steps)
    alarm = np.empty(steps, dtype=bool)
    attacked = np.zeros(steps, dtype=bool)

    u_prev = np.zeros(uav.l)
    for k in range(steps):
        x_true = uav.step(x_true, U[k])
        d = attack(k) if attack is not None else None
        attacked[k] = d is not None
        y = uav.measure(x_true, d=d)
        s = kf.step(u_prev, y)
        res = detector.step(s.innovation, s.S)
        u_prev = U[k]
        stat[k] = res.statistic
        alarm[k] = res.alarm

    first_attacked = int(np.argmax(attacked)) if attacked.any() else steps
    return DetectionRun(t, stat, alarm, attacked, detector.threshold, alpha,
                        burn_in, first_attacked)
