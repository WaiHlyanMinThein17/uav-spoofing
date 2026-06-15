"""Shared closed-loop runner for the Stage 2 (MPC) experiments.

All three Stage 2 experiments analyze the SAME closed-loop system: the UAV is
driven from start to goal by the MPC, using the Kalman filter's state estimate
as feedback (output-feedback MPC), under process and GPS measurement noise but
NO attack. Centralizing the run keeps the experiments independently runnable yet
guarantees they describe one consistent system.

This is the integrated estimation+control stack that later stages reuse: true
dynamics -> noisy GPS -> Kalman filter -> MPC on the estimate -> applied control.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.common import NavScenario
from uav_spoof.control.mpc import MPCController
from uav_spoof.estimation.kalman import KalmanFilter
from uav_spoof.simulation.dynamics import LinearUAV


@dataclass
class ClosedLoopRun:
    t: np.ndarray              # (T,) time
    true: np.ndarray           # (T, 4) true states
    est: np.ndarray            # (T, 4) KF estimates used as feedback
    controls: np.ndarray       # (T, 2) applied controls u_k
    initial_plan: np.ndarray   # (N+1, 4) MPC plan from the very first solve
    goal_state: np.ndarray     # (4,) goal
    scenario: NavScenario


def build_controller(uav: LinearUAV, sc: NavScenario) -> MPCController:
    """Build an MPC controller for a scenario.

    The controller depends only on the scenario (A, B, weights, bounds), not on
    the random seed, so a single instance can be reused across all Monte Carlo
    trials. This avoids recompiling the cvxpy problem once per trial -- the
    dominant cost in multi-trial evaluation.
    """
    return MPCController(
        uav,
        horizon=sc.horizon,
        Q=sc.Q(),
        R=sc.R(),
        Qf=sc.Qf(),
        state_bounds=sc.state_bounds(),
        control_bounds=sc.control_bounds(),
    )


def run_closed_loop(
    seed: int,
    scenario: NavScenario | None = None,
    mpc: MPCController | None = None,
) -> ClosedLoopRun:
    """Run output-feedback MPC navigation to the goal under noise, no attack.

    Args:
        seed: RNG seed for this trial's process/measurement noise.
        scenario: Navigation scenario (defaults to the canonical one).
        mpc: Optional prebuilt controller to reuse across trials. If None, one is
            built for this run. The controller is seed-independent, so reuse is
            safe and much faster.
    """
    sc = scenario or NavScenario()
    rng = np.random.default_rng(seed)
    uav = LinearUAV(dt=sc.dt, rng=rng)

    if mpc is None:
        mpc = build_controller(uav, sc)

    goal = sc.goal_state()
    x_true = sc.start_state()

    # Filter starts at the (known) start state with modest uncertainty.
    kf = KalmanFilter(uav, x0=sc.start_state(), P0=np.diag([0.5, 0.5, 0.2, 0.2]))

    T = sc.sim_steps
    t = np.arange(T) * sc.dt
    true_hist = np.empty((T, 4))
    est_hist = np.empty((T, 4))
    ctrl_hist = np.empty((T, 2))

    initial_plan = None
    u_prev = np.zeros(uav.l)
    for k in range(T):
        # Measure current true state, update estimate, then plan from estimate.
        y = uav.measure(x_true)               # no attack
        s = kf.step(u_prev, y)
        est = s.x_upd

        sol = mpc.solve(est, goal)
        if initial_plan is None:
            initial_plan = sol.planned_states
        u = sol.u0

        # Record, then advance the true system with the applied control.
        true_hist[k] = x_true
        est_hist[k] = est
        ctrl_hist[k] = u

        x_true = uav.step(x_true, u)
        u_prev = u

    return ClosedLoopRun(t, true_hist, est_hist, ctrl_hist, initial_plan, goal, sc)
