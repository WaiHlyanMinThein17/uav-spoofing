"""Stage 3 / Experiment 2: Detection rate against a naive attacker.

Research question:
    Does the chi-square detector catch a naive GPS spoofer that injects a sudden
    large constant position offset?

Claim supported:
    The sudden onset of a large bias produces a large innovation, so the detector
    flags the attack essentially immediately (per-trial detection rate). However,
    a *constant* bias is gradually absorbed by the Kalman filter -- the innovation
    returns toward normal -- so the *per-sample* alarm rate across the whole
    attack window is far lower than 1. This absorption is exactly why a stealthy,
    gradual attacker (Stage 4) can hope to evade the same detector.

Metrics reported (over N trials):
    * detection rate    -- fraction of trials in which the attack is caught at all
      (>=1 alarm during the attack window). 0/1 outcome -> Wilson CI.
    * steps to first detection after attack onset.
    * per-sample alarm rate during the attack (shows filter absorption).
    * pre-attack alarm rate (sanity: should match alpha).

Methodology:
    * DEVELOPMENT figure: one fixed seed -> nis_attack.png (NIS, threshold,
      attack onset).
    * REPORTED metrics: aggregated over N independent trials.

Outputs (results/stage3_detection/detection_rate/):
    nis_attack.png, metrics.json, trials.csv
"""

from __future__ import annotations

import numpy as np

from experiments.common import (DEV_SEEDS, MC_ENTROPY, N_TRIALS,
                                experiment_dir, save_metrics)
from experiments.mc import run_monte_carlo, save_trials_csv
from experiments.stage3_detection._runner import (DetectionRun,
                                                  naive_offset_attack,
                                                  run_detection)
from uav_spoof.visualization.plotting import new_axes, save_figure

STAGE, EXPERIMENT, KEY = "stage3_detection", "detection_rate", "stage3_detection"
ALPHA = 0.05
ATTACK_OFFSET = (8.0, 8.0)   # meters of injected GPS bias (naive: large + sudden)
ATTACK_START = 200           # step at which the attack begins
STEPS = 400


def _run(seed: int) -> DetectionRun:
    attack = naive_offset_attack(np.array(ATTACK_OFFSET), ATTACK_START)
    return run_detection(seed=seed, steps=STEPS, alpha=ALPHA,
                         attack=attack, attack_start=ATTACK_START)


def metrics_from_run(run: DetectionRun) -> dict[str, float]:
    s = run.attack_start
    attack_window = run.alarm[s:]
    caught = bool(attack_window.any())
    post = np.where(attack_window)[0]
    # If never caught, record the full window length as a censored time.
    ttd = float(post[0]) if post.size else float(attack_window.size)
    pre = run.alarm[run.burn_in:s]
    return {
        "detected": float(caught),                              # 0/1 outcome
        "steps_to_first_detection": ttd,
        "per_sample_alarm_rate_during_attack": float(attack_window.mean()),
        "pre_attack_alarm_rate": float(pre.mean()) if pre.size else 0.0,
    }


def make_figure(outdir) -> None:
    run = _run(DEV_SEEDS[KEY])  # dev seed, figure only
    fig, ax = new_axes("Chi-square detector vs naive large-offset attack",
                       "time (s)", r"NIS  $q_k=\nu^T S^{-1}\nu$")
    ax.plot(run.t, run.statistic, color="C2", lw=0.8, label="NIS")
    ax.axhline(run.threshold, color="C3", ls="--", lw=1.2,
               label=fr"threshold $\tau$ ($\alpha$={ALPHA})")
    ax.axvline(run.t[run.attack_start], color="black", ls=":", lw=1.2,
               label="attack onset")
    ax.set_yscale("log")  # onset spike dwarfs the rest; log keeps both visible
    ax.legend()
    save_figure(fig, outdir / "nis_attack.png")


def main() -> None:
    outdir = experiment_dir(STAGE, EXPERIMENT)
    make_figure(outdir)

    mc = run_monte_carlo(
        lambda seed: metrics_from_run(_run(seed)),
        base_entropy=MC_ENTROPY[KEY],
        n_trials=N_TRIALS,
        proportion_keys=["detected"],
    )
    save_trials_csv(outdir, mc)
    save_metrics(outdir, {
        "experiment": f"{STAGE}/{EXPERIMENT}",
        "research_question": "Does the detector catch a naive large-offset spoofer?",
        "alpha": ALPHA,
        "attack": {"type": "naive_constant_offset",
                   "offset_m": list(ATTACK_OFFSET),
                   "attack_start_step": ATTACK_START, "steps": STEPS},
        "evaluation": {"type": "monte_carlo", "n_trials": N_TRIALS,
                       "base_entropy": MC_ENTROPY[KEY],
                       "dev_seed_for_figure": DEV_SEEDS[KEY]},
        "metrics": mc.aggregates_to_dict(),
        "note": ("Per-trial detection is near-certain because the sudden bias "
                 "spikes the innovation at onset. The per-sample alarm rate "
                 "during the attack is much lower because the Kalman filter "
                 "absorbs a constant bias, returning the innovation toward "
                 "normal -- the opening that a gradual evasive attacker (Stage 4) "
                 "exploits."),
    })
    det = mc.aggregates["detected"]
    ttd = mc.aggregates["steps_to_first_detection"]
    ps = mc.aggregates["per_sample_alarm_rate_during_attack"]
    print(f"[{STAGE}/{EXPERIMENT}] over {N_TRIALS} trials: "
          f"detection rate={100*det.mean:.0f}% "
          f"(Wilson 95% CI [{100*det.ci95_low:.0f}, {100*det.ci95_high:.0f}]); "
          f"steps-to-detect mean={ttd.mean:.2f}; "
          f"per-sample alarm during attack={ps.mean:.3f}")


if __name__ == "__main__":
    main()
