"""Stage 2 / Experiment 1: Trajectory-following performance.

Research question:
    Does the MPC drive the UAV from start to the goal under output feedback?

Methodology:
    * DEVELOPMENT figure: one fixed seed -> trajectory.png.
    * REPORTED metrics: final position error, final speed, closest approach, and
      the fraction of trials that reach the settle radius -- aggregated over N
      independent trials. "reached_settle_radius" is a 0/1 outcome -> Wilson CI.

Outputs (results/stage2_mpc/trajectory_following/):
    trajectory.png, metrics.json, trials.csv
"""

from __future__ import annotations

import numpy as np

from experiments.common import (DEV_SEEDS, MC_ENTROPY, N_TRIALS, NavScenario,
                                experiment_dir, save_metrics, scenario_to_dict)
from experiments.mc import run_monte_carlo, save_trials_csv
from experiments.stage2_mpc._runner import (ClosedLoopRun, build_controller,
                                            run_closed_loop)
from uav_spoof.simulation.dynamics import LinearUAV
from uav_spoof.visualization.plotting import new_axes, save_figure

STAGE, EXPERIMENT, KEY = "stage2_mpc", "trajectory_following", "stage2_following"
SETTLE_RADIUS = 0.25  # meters


def metrics_from_run(run: ClosedLoopRun) -> dict[str, float]:
    goal_xy = np.array(run.scenario.goal_pos)
    dist = np.linalg.norm(run.true[:, :2] - goal_xy, axis=1)
    reached = bool(np.any(dist < SETTLE_RADIUS))
    return {
        "final_position_error_m": float(dist[-1]),
        "final_speed_mps": float(np.linalg.norm(run.true[-1, 2:])),
        "min_distance_to_goal_m": float(dist.min()),
        "reached_settle_radius": float(reached),  # 0/1 outcome
    }


def make_figure(outdir, sc: NavScenario, mpc) -> None:
    run = run_closed_loop(seed=DEV_SEEDS[KEY], scenario=sc, mpc=mpc)  # dev seed
    fig, ax = new_axes("MPC trajectory following (output feedback, no attack)",
                       "x (m)", "y (m)")
    ax.plot(run.initial_plan[:, 0], run.initial_plan[:, 1], color="C1", ls="--",
            label="initial MPC plan")
    ax.plot(run.true[:, 0], run.true[:, 1], color="black", label="realized path")
    ax.scatter(*sc.start_pos, color="C2", s=70, marker="o", label="start", zorder=5)
    ax.scatter(*sc.goal_pos, color="C3", s=110, marker="*", label="goal", zorder=5)
    ax.axis("equal")
    ax.legend()
    save_figure(fig, outdir / "trajectory.png")


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
        proportion_keys=["reached_settle_radius"],
    )
    save_trials_csv(outdir, mc)
    save_metrics(outdir, {
        "experiment": f"{STAGE}/{EXPERIMENT}",
        "research_question": "Does the MPC drive the UAV to the goal under feedback?",
        "settle_radius_m": SETTLE_RADIUS,
        "evaluation": {"type": "monte_carlo", "n_trials": N_TRIALS,
                       "base_entropy": MC_ENTROPY[KEY],
                       "dev_seed_for_figure": DEV_SEEDS[KEY]},
        "metrics": mc.aggregates_to_dict(),
        "scenario": scenario_to_dict(sc),
    })
    fe, reach = mc.aggregates["final_position_error_m"], mc.aggregates["reached_settle_radius"]
    print(f"[{STAGE}/{EXPERIMENT}] over {N_TRIALS} trials: "
          f"final err mean={fe.mean:.4f} std={fe.std:.4f} "
          f"min={fe.min:.4f} max={fe.max:.4f} m; "
          f"reached {SETTLE_RADIUS} m in {100*reach.mean:.0f}% of trials "
          f"(Wilson 95% CI [{100*reach.ci95_low:.0f}, {100*reach.ci95_high:.0f}])")


if __name__ == "__main__":
    main()
