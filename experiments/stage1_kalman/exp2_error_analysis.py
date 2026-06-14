"""Stage 1 / Experiment 2: Position error analysis.

Research question:
    How much does the Kalman filter reduce position error versus raw GPS?

Methodology:
    * DEVELOPMENT figure: one fixed seed -> position_error.png.
    * REPORTED metrics: raw GPS RMSE, KF RMSE, percent improvement, aggregated
      over N independent trials (mean / std / min / max / 95% CI).

Outputs (results/stage1_kalman/error_analysis/):
    position_error.png, metrics.json, trials.csv
"""

from __future__ import annotations

import numpy as np

from experiments.common import (DEV_SEEDS, MC_ENTROPY, N_TRIALS,
                                experiment_dir, save_metrics)
from experiments.mc import run_monte_carlo, save_trials_csv
from experiments.stage1_kalman._runner import TrackingRun, run_tracking
from uav_spoof.visualization.plotting import new_axes, save_figure

STAGE, EXPERIMENT, KEY = "stage1_kalman", "error_analysis", "stage1_error"


def metrics_from_run(run: TrackingRun) -> dict[str, float]:
    """Steady-state RMSEs (filtered vs raw GPS) and the percent improvement."""
    b = run.burn_in
    err_filt = np.linalg.norm(run.true[:, :2] - run.est[:, :2], axis=1)
    err_meas = np.linalg.norm(run.true[:, :2] - run.meas, axis=1)
    rmse_filt = float(np.sqrt(np.mean(err_filt[b:] ** 2)))
    rmse_meas = float(np.sqrt(np.mean(err_meas[b:] ** 2)))
    return {
        "raw_gps_rmse_m": rmse_meas,
        "kf_rmse_m": rmse_filt,
        "percent_improvement": float(100.0 * (1.0 - rmse_filt / rmse_meas)),
    }


def make_figure(outdir) -> None:
    run = run_tracking(seed=DEV_SEEDS[KEY])  # development seed, figure only
    b = run.burn_in
    err_filt = np.linalg.norm(run.true[:, :2] - run.est[:, :2], axis=1)
    err_meas = np.linalg.norm(run.true[:, :2] - run.meas, axis=1)
    fig, ax = new_axes("Position error: KF estimate vs raw GPS (no attack)",
                       "time (s)", "position error (m)")
    ax.plot(run.t, err_meas, color="C3", alpha=0.5, lw=0.9, label="raw GPS error")
    ax.plot(run.t, err_filt, color="C0", label="KF estimate error")
    ax.axvline(run.t[b], color="gray", ls=":", lw=1, label="burn-in cutoff")
    ax.legend()
    save_figure(fig, outdir / "position_error.png")


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
        "research_question": "How much does the KF reduce position error vs raw GPS?",
        "evaluation": {"type": "monte_carlo", "n_trials": N_TRIALS,
                       "base_entropy": MC_ENTROPY[KEY],
                       "dev_seed_for_figure": DEV_SEEDS[KEY]},
        "metrics": mc.aggregates_to_dict(),
    })
    raw, kf, imp = (mc.aggregates[k] for k in
                    ("raw_gps_rmse_m", "kf_rmse_m", "percent_improvement"))
    print(f"[{STAGE}/{EXPERIMENT}] over {N_TRIALS} trials: "
          f"raw={raw.mean:.4f}±{raw.std:.4f}  kf={kf.mean:.4f}±{kf.std:.4f}  "
          f"improvement={imp.mean:.1f}% (95% CI [{imp.ci95_low:.1f}, {imp.ci95_high:.1f}])")


if __name__ == "__main__":
    main()
