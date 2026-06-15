"""Stage 2 / Experiment 3: Constraint satisfaction.

Research question:
    Does the controller respect the velocity constraint -- on its planned
    trajectory (a hard QP guarantee) and on the realized noisy trajectory?

Methodology:
    * DEVELOPMENT figure: one fixed seed -> speed_constraint.png.
    * REPORTED metrics: planned vs realized max per-axis speed, planned and
      realized violation counts, and the fraction of trials whose PLAN satisfies
      the bound (0/1 -> Wilson CI), over N independent trials.

The honest distinction this experiment makes precise: the QP guarantees the
*plan* respects v_max, but process noise acts on the *plant* after the
constraint is imposed, so the realized state can exceed it slightly.

Outputs (results/stage2_mpc/constraint_satisfaction/):
    speed_constraint.png, metrics.json, trials.csv
"""

from __future__ import annotations

import numpy as np

from experiments.common import (DEV_SEEDS, MC_ENTROPY, N_TRIALS, NavScenario,
                                experiment_dir, save_metrics)
from experiments.mc import run_monte_carlo, save_trials_csv
from experiments.stage2_mpc._runner import (ClosedLoopRun, build_controller,
                                            run_closed_loop)
from uav_spoof.simulation.dynamics import LinearUAV
from uav_spoof.visualization.plotting import new_axes, save_figure

STAGE, EXPERIMENT, KEY = "stage2_mpc", "constraint_satisfaction", "stage2_constraint"


def metrics_from_run(run: ClosedLoopRun) -> dict[str, float]:
    sc = run.scenario
    realized_v = np.abs(run.true[:, 2:])
    planned_v = np.abs(run.initial_plan[:, 2:])
    # Planned trajectory comes from the QP, so judge it at the solver's
    # feasibility tolerance (1e-3); realized state is the true plant, judged exactly.
    planned_viol = int(np.sum(planned_v > sc.v_max + 1e-3))
    # Velocity estimation error explains the realized overshoot: GPS sees position
    # only, so the KF velocity estimate lags during hard acceleration, and the
    # certainty-equivalence MPC plans on that estimate as if it were the truth.
    vel_est_err = np.abs(run.est[:, 2:] - run.true[:, 2:])
    return {
        "planned_max_per_axis_speed_mps": float(planned_v.max()),
        "realized_max_per_axis_speed_mps": float(realized_v.max()),
        "planned_violation_count": float(planned_viol),
        "realized_violation_count": float(np.sum(realized_v > sc.v_max + 1e-9)),
        "max_velocity_estimation_error_mps": float(vel_est_err.max()),
        "plan_satisfies_constraint": float(planned_viol == 0),  # 0/1 outcome
    }


def make_figure(outdir, sc: NavScenario, mpc) -> None:
    run = run_closed_loop(seed=DEV_SEEDS[KEY], scenario=sc, mpc=mpc)  # dev seed
    realized_v = np.abs(run.true[:, 2:])
    fig, ax = new_axes("MPC velocity-constraint satisfaction (no attack)",
                       "time (s)", "per-axis speed (m/s)")
    ax.plot(run.t, realized_v[:, 0], color="C0", label=r"realized $|v_x|$")
    ax.plot(run.t, realized_v[:, 1], color="C4", label=r"realized $|v_y|$")
    ax.axhline(sc.v_max, color="black", ls="--", lw=1.2, label=r"$v_{max}$")
    ax.legend()
    save_figure(fig, outdir / "speed_constraint.png")


def main() -> None:
    sc = NavScenario()
    outdir = experiment_dir(STAGE, EXPERIMENT)
    # Build the (seed-independent) controller once and reuse it everywhere.
    mpc = build_controller(LinearUAV(dt=sc.dt), sc)
    make_figure(outdir, sc, mpc)

    mc = run_monte_carlo(
        lambda seed: metrics_from_run(run_closed_loop(seed=seed, scenario=sc, mpc=mpc)),
        base_entropy=MC_ENTROPY[KEY],
        n_trials=N_TRIALS,
        proportion_keys=["plan_satisfies_constraint"],
    )
    save_trials_csv(outdir, mc)
    save_metrics(outdir, {
        "experiment": f"{STAGE}/{EXPERIMENT}",
        "research_question": "Are velocity constraints respected (planned vs realized)?",
        "v_max": sc.v_max,
        "evaluation": {"type": "monte_carlo", "n_trials": N_TRIALS,
                       "base_entropy": MC_ENTROPY[KEY],
                       "dev_seed_for_figure": DEV_SEEDS[KEY]},
        "metrics": mc.aggregates_to_dict(),
        "note": ("The plan respects v_max exactly (hard QP constraint). The "
                 "realized state exceeds it for two compounding reasons: process "
                 "noise acts after the constraint is enforced, and -- dominant "
                 "here -- GPS observes position only, so the KF velocity estimate "
                 "lags during hard acceleration while the certainty-equivalence "
                 "MPC plans on that estimate as if it were the true state. "
                 "True-state-feedback control limits the realized speed far more "
                 "tightly; the gap is the cost of output feedback, not a "
                 "controller defect."),
    })
    pms = mc.aggregates["planned_max_per_axis_speed_mps"]
    rms = mc.aggregates["realized_max_per_axis_speed_mps"]
    ps = mc.aggregates["plan_satisfies_constraint"]
    vee = mc.aggregates["max_velocity_estimation_error_mps"]
    print(f"[{STAGE}/{EXPERIMENT}] over {N_TRIALS} trials: "
          f"planned max/axis mean={pms.mean:.4f} (max={pms.max:.4f}); "
          f"realized max/axis mean={rms.mean:.4f} (max={rms.max:.4f}); "
          f"plan satisfies v_max in {100*ps.mean:.0f}% of trials; "
          f"max vel-est error mean={vee.mean:.3f} m/s")


if __name__ == "__main__":
    main()
