"""Model Predictive Control for UAV trajectory planning (the 'control' layer).

At each timestep the controller solves a constrained finite-horizon QP and
applies only the first control (receding horizon):

    min_{x,u}  sum_{i=1}^{N-1} (x_i - g)^T Q (x_i - g)
               + (x_N - g)^T Qf (x_N - g)
               + sum_{i=0}^{N-1} u_i^T R u_i
    s.t.       x_0 = x_init                         (current state estimate)
               x_{i+1} = A x_i + B u_i              (dynamics)
               x_lb <= x_i <= x_ub,  i = 1..N       (state box; NOT on x_0)
               u_lb <= u_i <= u_ub,  i = 0..N-1     (control box)

This is a convex QP (PSD quadratic cost, linear equality/inequality
constraints), so any local optimum is the unique global optimum.

Feasibility strategy (hard primary + soft fallback)
---------------------------------------------------
The state box is HARD by default. Hard constraints are what actually limit the
*realized* closed-loop velocity: because the plan cannot exceed v_max at ANY
horizon step (including x_1), the applied control is forced to brake in time, so
the realized speed stays at v_max plus only process noise. A purely soft bound,
by contrast, lets the receding-horizon planner keep deferring the braking step
and the realized speed creeps well past the limit.

However, under an unlucky noise draw the estimated velocity can exceed
v_max by more than one step of bounded deceleration can remove, making the HARD
x_1 box infeasible. So the controller keeps a SOFT twin of the problem (state box
relaxed by nonnegative slack s with a large L1 penalty) and falls back to it only
when the hard solve is infeasible. In the common case the hard solve is used
(exact binding, tight limiting); the soft fallback merely guarantees a usable
command always exists. The control box is always HARD -- actuator limits are
physical.

Costs use Cholesky factors with sum_squares (||L^T (x-g)||^2 = (x-g)^T Q (x-g))
instead of quad_form, so the problem is DPP-compliant with parametric x_init and
goal, letting OSQP reuse its factorization across timesteps.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from uav_spoof.simulation.dynamics import LinearUAV


@dataclass
class MPCSolution:
    """Result of one MPC solve."""

    u0: np.ndarray                # control to apply now (l,)
    planned_states: np.ndarray    # (N+1, n) predicted state trajectory
    planned_controls: np.ndarray  # (N, l) predicted control sequence
    status: str                   # solver status string
    objective: float              # optimal objective value
    used_soft_fallback: bool      # True if the hard QP was infeasible this step
    max_slack: float              # state-box slack used (0 unless fallback fired)


class _SubProblem:
    """One cvxpy QP instance (hard or soft) sharing external parameters."""

    def __init__(self, prob, X, U, S):
        self.prob, self.X, self.U, self.S = prob, X, U, S


class MPCController:
    """Receding-horizon QP controller with hard-primary / soft-fallback solves."""

    def __init__(
        self,
        uav: LinearUAV,
        horizon: int,
        Q: np.ndarray,
        R: np.ndarray,
        Qf: np.ndarray,
        state_bounds: tuple[np.ndarray, np.ndarray],
        control_bounds: tuple[np.ndarray, np.ndarray],
        slack_penalty: float = 1.0e5,
        solver: str = cp.OSQP,
        solver_opts: dict | None = None,
    ) -> None:
        self.A, self.B = uav.A, uav.B
        self.n, self.l = uav.n, uav.l
        self.N = horizon
        self.solver = solver
        # OSQP tolerances chosen for Monte Carlo throughput (hundreds of trials x
        # ~130 solves). At eps=1e-5 the planned trajectory binds the hard state
        # box to <3e-4, well inside the 1e-3 feasibility tolerance the
        # experiments judge against, while each solve stays a few milliseconds.
        self.solver_opts = solver_opts or {
            "eps_abs": 1e-5, "eps_rel": 1e-5, "max_iter": 6000, "polishing": False,
        }

        self._Lq = np.linalg.cholesky(Q + 1e-12 * np.eye(self.n))
        self._Lqf = np.linalg.cholesky(Qf + 1e-12 * np.eye(self.n))
        self._Lr = np.linalg.cholesky(R + 1e-12 * np.eye(self.l))
        self._xb = state_bounds
        self._ub = control_bounds
        self._slack_penalty = slack_penalty

        # Parameters shared by both sub-problems.
        self._x_init = cp.Parameter(self.n, name="x_init")
        self._goal = cp.Parameter(self.n, name="goal")

        self._hard = self._build(soft=False)
        self._soft = self._build(soft=True)

    def _build(self, soft: bool) -> _SubProblem:
        x_lb, x_ub = self._xb
        u_lb, u_ub = self._ub
        X = cp.Variable((self.n, self.N + 1))
        U = cp.Variable((self.l, self.N))
        S = cp.Variable((self.n, self.N), nonneg=True) if soft else None

        cost = 0
        cons = [X[:, 0] == self._x_init]
        for i in range(self.N):
            cons += [X[:, i + 1] == self.A @ X[:, i] + self.B @ U[:, i]]
            cons += [U[:, i] >= u_lb, U[:, i] <= u_ub]      # hard control box
            cost += cp.sum_squares(self._Lr.T @ U[:, i])
            if i >= 1:
                if soft:
                    s = S[:, i - 1]
                    cons += [X[:, i] >= x_lb - s, X[:, i] <= x_ub + s]
                else:
                    cons += [X[:, i] >= x_lb, X[:, i] <= x_ub]
                cost += cp.sum_squares(self._Lq.T @ (X[:, i] - self._goal))
        if soft:
            sT = S[:, self.N - 1]
            cons += [X[:, self.N] >= x_lb - sT, X[:, self.N] <= x_ub + sT]
            cost += self._slack_penalty * cp.sum(S)
        else:
            cons += [X[:, self.N] >= x_lb, X[:, self.N] <= x_ub]
        cost += cp.sum_squares(self._Lqf.T @ (X[:, self.N] - self._goal))

        return _SubProblem(cp.Problem(cp.Minimize(cost), cons), X, U, S)

    def solve(self, x_current: np.ndarray, goal: np.ndarray) -> MPCSolution:
        """Solve for the current state; try hard first, fall back to soft."""
        self._x_init.value = np.asarray(x_current, dtype=float)
        self._goal.value = np.asarray(goal, dtype=float)

        self._hard.prob.solve(solver=self.solver, warm_start=True, **self.solver_opts)
        sub, used_soft = self._hard, False
        if self._hard.X.value is None or self._hard.prob.status not in (
                "optimal", "optimal_inaccurate"):
            # Rare: hard QP infeasible for this noise draw -> soft fallback.
            self._soft.prob.solve(solver=self.solver, warm_start=True, **self.solver_opts)
            sub, used_soft = self._soft, True
            if self._soft.X.value is None:
                raise RuntimeError(
                    f"MPC soft fallback failed: {self._soft.prob.status}")

        max_slack = float(sub.S.value.max()) if (used_soft and sub.S.value is not None) else 0.0
        return MPCSolution(
            u0=sub.U.value[:, 0].copy(),
            planned_states=sub.X.value.T.copy(),
            planned_controls=sub.U.value.T.copy(),
            status=sub.prob.status,
            objective=float(sub.prob.value),
            used_soft_fallback=used_soft,
            max_slack=max_slack,
        )
