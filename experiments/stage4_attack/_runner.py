"""Shared runner for the Stage 4 (evasive attacker) experiments.

This is the full integrated system: true dynamics -> (optionally spoofed) GPS ->
Kalman filter -> chi-square detector -> MPC on the estimate -> applied control,
with the evasive attacker injecting a stealthy offset into the measurement.

To measure attack-induced deviation cleanly, each call runs TWO passes on the
SAME pre-drawn noise sequences:
  * nominal  -- no attack (the trajectory that would have happened),
  * attacked -- the evasive attacker active from its start step.
Because both passes use identical process/measurement noise, any difference
between the two true trajectories is attributable solely to the attack.

The detector is the unchanged Stage 3 ChiSquareDetector; the attacker is the
Stage 4 EvasiveAttacker. The two never share logic -- the runner wires them
together but keeps attack optimization and detection strictly separate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.common import NavScenario
from uav_spoof.attack.evasive import AttackerConfig, EvasiveAttacker
from uav_spoof.control.mpc import MPCController
from uav_spoof.detection.chi_square import ChiSquareDetector
from uav_spoof.estimation.kalman import KalmanFilter
from uav_spoof.simulation.dynamics import LinearUAV


@dataclass
class AttackRun:
    t: np.ndarray              # (T,)
    true_nominal: np.ndarray   # (T,4) true states, no attack
    true_attacked: np.ndarray  # (T,4) true states under attack
    est_attacked: np.ndarray   # (T,4) filter estimates under attack
    statistic: np.ndarray      # (T,) detector NIS under attack
    alarm: np.ndarray          # (T,) bool alarms under attack
    offsets: np.ndarray        # (T,2) injected spoof offsets
    attacked_mask: np.ndarray  # (T,) bool: step is within the attack window
    threshold: float           # detector threshold tau
    legit_goal: np.ndarray     # (4,)
    attacker_goal: np.ndarray  # (2,) position
    attack_start: int
    scenario: NavScenario


def build_controller(uav: LinearUAV, sc: NavScenario) -> MPCController:
    # Fast solver profile: Stage 4 runs the closed loop twice per trial plus a
    # per-step SLSQP attacker solve, so MPC speed dominates. eps=1e-4 is ample
    # here (the attack experiments do not probe constraint-binding precision).
    fast = {"eps_abs": 1e-4, "eps_rel": 1e-4, "max_iter": 2000, "polishing": False}
    return MPCController(uav, horizon=sc.horizon, Q=sc.Q(), R=sc.R(), Qf=sc.Qf(),
                         state_bounds=sc.state_bounds(),
                         control_bounds=sc.control_bounds(), solver_opts=fast)


def persistence_detection(alarm: np.ndarray, start: int, window: int = 10,
                          k_of_n: int = 4) -> int:
    """First step >= start at which >= k_of_n alarms occur within `window` steps.

    A debounce rule converting the per-sample chi-square alarms into a robust
    per-trial detection decision: the per-sample detector is UNCHANGED; this is an
    evaluation-side monitor. Returns -1 if never triggered.

    Calibration matters here. The rule is applied over the full ~100-step attack
    window, and a loose rule false-triggers on baseline noise: measured over the
    no-attack window, >=3-in-10 fires 21% of the time (uselessly high), whereas
    >=4-in-10 fires only 2.5% of the time. We therefore use >=4-in-10, whose
    no-attack floor (~0.025) sits at/below the per-sample alpha=0.05, so a
    reported detection rate reflects the attack rather than the monitor's own
    false alarms.
    """
    for i in range(start, len(alarm)):
        lo = max(0, i - window + 1)
        if alarm[lo:i + 1].sum() >= k_of_n:
            return i
    return -1


def run_attack(
    seed: int,
    attacker_goal,
    attack_start: int = 30,
    d_max: float = 2.0,
    delta_max: float = 0.3,
    alpha: float = 0.05,
    scenario: NavScenario | None = None,
    mpc: MPCController | None = None,
    with_nominal: bool = True,
) -> AttackRun:
    """Run the attacked closed loop (and, if with_nominal, a same-noise nominal).

    with_nominal=False skips the no-attack pass (used by experiments that don't
    need the nominal trajectory, e.g. attack success / stealthiness / tradeoff),
    roughly halving runtime.
    """
    sc = scenario or NavScenario()
    uav = LinearUAV(dt=sc.dt)
    A, B, C = uav.A, uav.B, uav.C
    m, l = uav.m, uav.l
    T = sc.sim_steps
    goal = sc.goal_state()

    if mpc is None:
        mpc = build_controller(uav, sc)
    detector = ChiSquareDetector(dof=m, alpha=alpha)

    # Pre-draw noise so both passes are driven by the SAME randomness.
    rng = np.random.default_rng(seed)
    W = rng.multivariate_normal(np.zeros(uav.n), uav.Sigma_w, size=T)
    V = rng.multivariate_normal(np.zeros(m), uav.Sigma_v, size=T)

    cfg = AttackerConfig(goal_pos=np.asarray(attacker_goal, float),
                         legit_goal_pos=np.array(sc.goal_pos, float),
                         attack_start=attack_start, d_max=d_max,
                         delta_max=delta_max, tau=detector.threshold)
    attacker = EvasiveAttacker(C, cfg)

    def simulate(use_attack: bool):
        x = sc.start_state()
        kf = KalmanFilter(uav, x0=sc.start_state(), P0=np.diag([0.5, 0.5, 0.2, 0.2]))
        true_h = np.empty((T, 4)); est_h = np.empty((T, 4))
        stat_h = np.empty(T); alarm_h = np.empty(T, dtype=bool)
        off_h = np.zeros((T, 2))
        u_prev = np.zeros(l); d_prev = np.zeros(m)
        for k in range(T):
            y0 = C @ x + V[k]                      # clean measurement (pre-drawn noise)
            if use_attack:
                x_pred, P_pred = kf.predict(u_prev)
                S, K = kf.gain(P_pred)
                nu0 = y0 - C @ x_pred
                d = attacker.compute(k, x_pred, S, K, nu0, x, d_prev)
            else:
                d = np.zeros(m)
            y = y0 + d
            s = kf.step(u_prev, y)
            det = detector.step(s.innovation, s.S)
            sol = mpc.solve(s.x_upd, goal)
            u = sol.u0

            true_h[k] = x; est_h[k] = s.x_upd
            stat_h[k] = det.statistic; alarm_h[k] = det.alarm
            off_h[k] = d
            x = A @ x + B @ u + W[k]
            u_prev = u; d_prev = d
        return true_h, est_h, stat_h, alarm_h, off_h

    true_atk, est_atk, stat_atk, alarm_atk, offsets = simulate(use_attack=True)
    if with_nominal:
        true_nom, _, _, _, _ = simulate(use_attack=False)
    else:
        true_nom = true_atk.copy()  # nominal not requested; deviation is undefined

    t = np.arange(T) * sc.dt
    attacked_mask = np.arange(T) >= attack_start
    return AttackRun(t, true_nom, true_atk, est_atk, stat_atk, alarm_atk, offsets,
                     attacked_mask, detector.threshold, goal,
                     np.asarray(attacker_goal, float), attack_start, sc)
