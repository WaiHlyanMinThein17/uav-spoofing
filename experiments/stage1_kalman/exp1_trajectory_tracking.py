"""Stage 1 / Experiment 1: Trajectory tracking.

Research question:
    Does the Kalman filter reconstruct the true UAV trajectory from noisy GPS?

Methodology:
    * DEVELOPMENT figure: one fixed seed (DEV_SEEDS) -> trajectory.png.
    * REPORTED metric: trajectory RMSE aggregated over N independent trials
      (mean / std / min / max / 95% CI), persisted to trials.csv + metrics.json.

Outputs (results/stage1_kalman/trajectory_tracking/):
    trajectory.png, metrics.json, trials.csv
"""

from __future__ import annotations

import numpy as np

from experiments.common import (DEV_SEEDS, MC_ENTROPY, N_TRIALS,
                                experiment_dir, save_metrics)
from experiments.mc import run_monte_carlo, save_trials_csv
from experiments.stage1_kalman._runner import TrackingRun, run_tracking
from uav_spoof.visualization.plotting import new_axes, save_figure

STAGE, EXPERIMENT, KEY = "stage1_kalman", "trajectory_tracking", "stage1_trajectory"


def metrics_from_run(run: TrackingRun) -> dict[str, float]:
    """Scalar metrics for one trial: full-trajectory position RMSE."""
    pos_err = np.linalg.norm(run.true[:, :2] - run.est[:, :2], axis=1)
    return {"trajectory_rmse_m": float(np.sqrt(np.mean(pos_err**2)))}


def make_figure(outdir) -> None:
    run = run_tracking(seed=DEV_SEEDS[KEY])  # development seed, figure only
    fig, ax = new_axes("Kalman filter trajectory tracking (no attack)",
                       "x (m)", "y (m)")
    ax.plot(run.true[:, 0], run.true[:, 1], color="black", lw=2, label="true")
    ax.scatter(run.meas[::6, 0], run.meas[::6, 1], s=8, color="C3", alpha=0.35,
               label="GPS measurements")
    ax.plot(run.est[:, 0], run.est[:, 1], color="C0", ls="--", label="KF estimate")
    ax.axis("equal")
    ax.legend()
    save_figure(fig, outdir / "trajectory.png")


def main() -> None:
    outdir = experiment_dir(STAGE, EXPERIMENT)
    make_figure(outdir)

    mc = run_monte_carlo(
        lambda seed: metrics_from_run(run_tracking(seed=seed)),
        base_entropy=MC_ENTROPY[KEY],
        n_trials=N_TRIALS,
    )
    save_trials_csv(outdir, mc)
    save_metrics(outdir, {
        "experiment": f"{STAGE}/{EXPERIMENT}",
        "research_question": "Does the KF reconstruct the true trajectory from noisy GPS?",
        "evaluation": {"type": "monte_carlo", "n_trials": N_TRIALS,
                       "base_entropy": MC_ENTROPY[KEY],
                       "dev_seed_for_figure": DEV_SEEDS[KEY]},
        "metrics": mc.aggregates_to_dict(),
    })
    a = mc.aggregates["trajectory_rmse_m"]
    print(f"[{STAGE}/{EXPERIMENT}] trajectory RMSE over {N_TRIALS} trials: "
          f"mean={a.mean:.4f} std={a.std:.4f} min={a.min:.4f} max={a.max:.4f} "
          f"95% CI=[{a.ci95_low:.4f}, {a.ci95_high:.4f}] m")


if __name__ == "__main__":
    main()
