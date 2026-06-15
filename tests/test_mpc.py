"""Tests for the MPC controller."""

import numpy as np

from experiments.common import NavScenario
from uav_spoof.control.mpc import MPCController
from uav_spoof.simulation.dynamics import LinearUAV


def _build():
    sc = NavScenario()
    uav = LinearUAV(dt=sc.dt, sigma_w_pos=0.0, sigma_w_vel=0.0, sigma_v=0.0)
    # Tests verify the FORMULATION, so use an accurate (polished) solver profile;
    # the experiments use a faster profile for Monte Carlo throughput.
    accurate = {"eps_abs": 1e-6, "eps_rel": 1e-6, "max_iter": 20000, "polishing": True}
    mpc = MPCController(uav, horizon=sc.horizon, Q=sc.Q(), R=sc.R(), Qf=sc.Qf(),
                       state_bounds=sc.state_bounds(),
                       control_bounds=sc.control_bounds(),
                       solver_opts=accurate)
    return sc, uav, mpc


def test_solve_is_optimal_and_control_within_bounds():
    sc, _, mpc = _build()
    sol = mpc.solve(sc.start_state(), sc.goal_state())
    assert sol.status in ("optimal", "optimal_inaccurate")
    # OSQP satisfies constraints to its tolerance; judge at 1e-3.
    assert np.all(np.abs(sol.u0) <= sc.a_max + 1e-3)


def test_planned_states_respect_box_constraints():
    sc, _, mpc = _build()
    sol = mpc.solve(sc.start_state(), sc.goal_state())
    lb, ub = sc.state_bounds()
    # From the noiseless start the hard problem is feasible, so slack should be
    # ~0 and x_1..x_N should respect the box at the solver tolerance.
    assert sol.max_slack < 1e-5
    assert np.all(sol.planned_states[1:] >= lb - 1e-5)
    assert np.all(sol.planned_states[1:] <= ub + 1e-5)


def test_closed_loop_reduces_distance_to_goal():
    # Noiseless true-state feedback: the controller must make progress to goal.
    sc, uav, mpc = _build()
    x = sc.start_state()
    goal_xy = np.array(sc.goal_pos)
    d0 = np.linalg.norm(x[:2] - goal_xy)
    for _ in range(60):
        sol = mpc.solve(x, sc.goal_state())
        x = uav.step(x, sol.u0)
    d_final = np.linalg.norm(x[:2] - goal_xy)
    assert d_final < 0.5 * d0
