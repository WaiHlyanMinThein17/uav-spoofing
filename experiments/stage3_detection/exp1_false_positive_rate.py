"""Stage 3 / Experiment 1: False-positive rate under no attack.

Research question:
    With no attacker present, how often does the chi-square detector raise a
    false alarm -- and does that rate match the significance level alpha?

Claim supported:
    Under no attack the per-sample false-positive rate equals alpha by
    construction, because NIS ~ chi-square(m) and the threshold is its (1-alpha)
    quantile. This is the direct consequence of the Stage 1 consistency result.

Methodology:
    * DEVELOPMENT figure: one fixed seed -> nis_no_attack.png (NIS vs threshold).
    * REPORTED metric: per-trial false-positive rate aggregated over N trials.

Outputs (results/stage3_detection/false_positive_rate/):
    nis_no_attack.png, metrics.json, trials.csv
"""

from __future__ import annotations

import numpy as np

from experiments.common import (DEV_SEEDS, MC_ENTROPY, N_TRIALS,
                                experiment_dir, save_metrics)
from experiments.mc import run_monte_carlo, save_trials_csv
from experiments.stage3_detection._runner import DetectionRun, run_detection
from uav_spoof.visualization.plotting import new_axes, save_figure

STAGE, EXPERIMENT, KEY = "stage3_detection", "false_positive_rate", "stage3_false_positive"
ALPHA = 0.05


def metrics_from_run(run: DetectionRun) -> dict[str, float]:
    """Per-trial false-positive rate: fraction of post-burn-in steps flagged."""
    alarms = run.alarm[run.burn_in:]
    return {"false_positive_rate": float(alarms.mean())}


def make_figure(outdir) -> None:
    run = run_detection(seed=DEV_SEEDS[KEY], alpha=ALPHA)  # dev seed, no attack
    fig, ax = new_axes("Chi-square detector under no attack",
                       "time (s)", r"NIS  $q_k=\nu^T S^{-1}\nu$")
    ax.plot(run.t, run.statistic, color="C2", lw=0.8, label="NIS")
    ax.axhline(run.threshold, color="C3", ls="--", lw=1.2,
               label=fr"threshold $\tau$ ($\alpha$={ALPHA})")
    flagged = run.alarm.copy()
    flagged[:run.burn_in] = False
    ax.scatter(run.t[flagged], run.statistic[flagged], s=12, color="C3",
               zorder=5, label="false alarm")
    ax.legend()
    save_figure(fig, outdir / "nis_no_attack.png")


def main() -> None:
    outdir = experiment_dir(STAGE, EXPERIMENT)
    make_figure(outdir)

    mc = run_monte_carlo(
        lambda seed: metrics_from_run(run_detection(seed=seed, alpha=ALPHA)),
        base_entropy=MC_ENTROPY[KEY],
        n_trials=N_TRIALS,
    )
    save_trials_csv(outdir, mc)
    save_metrics(outdir, {
        "experiment": f"{STAGE}/{EXPERIMENT}",
        "research_question": "What is the detector false-positive rate under no attack?",
        "alpha": ALPHA,
        "evaluation": {"type": "monte_carlo", "n_trials": N_TRIALS,
                       "base_entropy": MC_ENTROPY[KEY],
                       "dev_seed_for_figure": DEV_SEEDS[KEY]},
        "metrics": mc.aggregates_to_dict(),
    })
    fpr = mc.aggregates["false_positive_rate"]
    print(f"[{STAGE}/{EXPERIMENT}] over {N_TRIALS} trials: "
          f"false-positive rate mean={fpr.mean:.4f} std={fpr.std:.4f} "
          f"(target alpha={ALPHA}); 95% CI [{fpr.ci95_low:.4f}, {fpr.ci95_high:.4f}]")


if __name__ == "__main__":
    main()
