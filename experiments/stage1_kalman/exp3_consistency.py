"""Stage 1 / Experiment 3: Statistical consistency.

Research question:
    Is the filter statistically consistent -- does its normalized innovation
    (NIS) behave like the chi-square(m) variable the model predicts?

Why this matters: the Stage 3 spoofing detector thresholds exactly this NIS
statistic. If NIS is well-calibrated under no attack, the detector's
false-positive rate equals the chosen significance level by construction.

Methodology:
    * DEVELOPMENT figure: one fixed seed -> nis.png (with the chi-square band).
    * REPORTED metrics: per-trial mean NIS and per-trial fraction-within-band,
      aggregated over N independent trials.

Outputs (results/stage1_kalman/consistency/):
    nis.png, metrics.json, trials.csv
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from experiments.common import (DEV_SEEDS, MC_ENTROPY, N_TRIALS,
                                experiment_dir, save_metrics)
from experiments.mc import run_monte_carlo, save_trials_csv
from experiments.stage1_kalman._runner import TrackingRun, run_tracking
from uav_spoof.visualization.plotting import new_axes, save_figure

STAGE, EXPERIMENT, KEY = "stage1_kalman", "consistency", "stage1_consistency"
DOF = 2          # measurement dimension m -> NIS degrees of freedom
ALPHA = 0.05     # two-sided 95% consistency band
BAND = (float(chi2.ppf(ALPHA / 2, DOF)), float(chi2.ppf(1 - ALPHA / 2, DOF)))


def metrics_from_run(run: TrackingRun) -> dict[str, float]:
    """Per-trial mean NIS and fraction of NIS samples inside the 95% band."""
    nis = run.nis[run.burn_in:]
    return {
        "mean_nis": float(nis.mean()),
        "fraction_within_95_band": float(np.mean((nis >= BAND[0]) & (nis <= BAND[1]))),
    }


def make_figure(outdir) -> None:
    run = run_tracking(seed=DEV_SEEDS[KEY])  # development seed, figure only
    fig, ax = new_axes("Kalman filter NIS consistency (no attack)",
                       "time (s)", r"NIS  $\nu^T S^{-1}\nu$")
    ax.plot(run.t, run.nis, color="C2", lw=0.8)
    ax.axhline(BAND[1], color="black", ls="--", lw=1, label=f"95% band (df={DOF})")
    ax.axhline(BAND[0], color="black", ls="--", lw=1)
    ax.axhline(DOF, color="C1", ls=":", lw=1.3, label=f"E[NIS]={DOF}")
    ax.legend()
    save_figure(fig, outdir / "nis.png")


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
        "research_question": "Is the filter statistically consistent (NIS ~ chi2(2))?",
        "dof": DOF, "expected_mean_nis": DOF,
        "band_95_lower": BAND[0], "band_95_upper": BAND[1],
        "evaluation": {"type": "monte_carlo", "n_trials": N_TRIALS,
                       "base_entropy": MC_ENTROPY[KEY],
                       "dev_seed_for_figure": DEV_SEEDS[KEY]},
        "metrics": mc.aggregates_to_dict(),
    })
    nis, frac = mc.aggregates["mean_nis"], mc.aggregates["fraction_within_95_band"]
    print(f"[{STAGE}/{EXPERIMENT}] over {N_TRIALS} trials: "
          f"mean NIS={nis.mean:.3f}±{nis.std:.3f} (expect {DOF}); "
          f"within-band={100*frac.mean:.1f}% "
          f"(95% CI [{100*frac.ci95_low:.1f}, {100*frac.ci95_high:.1f}])")


if __name__ == "__main__":
    main()
