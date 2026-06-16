"""Stage 4 / Experiment 2: Stealthiness.

Research question:
    How well does the evasive attacker stay below the chi-square detector while
    redirecting the UAV?

Methodology:
    * DEVELOPMENT figure: one seed -> detector_statistic.png (NIS vs threshold,
      attack onset marked).
    * REPORTED metrics over N trials:
        - alarm rate during the attack window (per-sample; compare to alpha),
        - detection rate (per-trial, via the persistence debounce rule),
        - time-to-detection after attack onset (censored to window if never).

The per-sample detector is the UNCHANGED Stage 3 detector. The persistence rule
(>=4 alarms within 10 steps) is an evaluation-side monitor used only to turn the
per-sample alarms into a robust per-trial detection decision.

Outputs (results/stage4_attack/stealthiness/):
    detector_statistic.png, metrics.json, trials.csv
"""

from __future__ import annotations

import numpy as np

from experiments.common import (ATTACK_ALPHA, ATTACK_D_MAX, ATTACK_DELTA_MAX,
                                ATTACK_GOAL, ATTACK_START, DEV_SEEDS, MC_ENTROPY,
                                N_TRIALS_STAGE4, NavScenario, experiment_dir,
                                save_metrics)
from experiments.mc import run_monte_carlo, save_trials_csv
from experiments.stage4_attack._runner import (AttackRun, build_controller,
                                               persistence_detection, run_attack)
from uav_spoof.simulation.dynamics import LinearUAV
from uav_spoof.visualization.plotting import new_axes, save_figure

STAGE, EXPERIMENT, KEY = "stage4_attack", "stealthiness", "stage4_stealth"


def metrics_from_run(run: AttackRun) -> dict[str, float]:
    window = run.alarm[run.attack_start:]
    det_idx = persistence_detection(run.alarm, run.attack_start)
    detected = det_idx != -1
    # time-to-detection in steps after onset; censored to window length if never.
    ttd = float(det_idx - run.attack_start) if detected else float(window.size)
    return {
        "alarm_rate_during_attack": float(window.mean()),
        "detected": float(detected),                 # 0/1 outcome
        "time_to_detection_steps": ttd,
    }


def make_figure(outdir, sc, mpc) -> None:
    run = run_attack(seed=DEV_SEEDS[KEY], attacker_goal=ATTACK_GOAL,
                     attack_start=ATTACK_START, d_max=ATTACK_D_MAX,
                     delta_max=ATTACK_DELTA_MAX, alpha=ATTACK_ALPHA,
                     scenario=sc, mpc=mpc, with_nominal=False)
    fig, ax = new_axes("Detector statistic under evasive attack",
                       "time (s)", r"NIS  $q_k=\nu^T S^{-1}\nu$")
    ax.plot(run.t, run.statistic, color="C2", lw=0.8, label="NIS")
    ax.axhline(run.threshold, color="C3", ls="--", lw=1.2,
               label=fr"threshold $\tau$ ($\alpha$={ATTACK_ALPHA})")
    ax.axvline(run.t[run.attack_start], color="black", ls=":", lw=1.2,
               label="attack onset")
    ax.legend()
    save_figure(fig, outdir / "detector_statistic.png")


def main() -> None:
    sc = NavScenario()
    outdir = experiment_dir(STAGE, EXPERIMENT)
    mpc = build_controller(LinearUAV(dt=sc.dt), sc)
    make_figure(outdir, sc, mpc)

    mc = run_monte_carlo(
        lambda seed: metrics_from_run(run_attack(
            seed=seed, attacker_goal=ATTACK_GOAL, attack_start=ATTACK_START,
            d_max=ATTACK_D_MAX, delta_max=ATTACK_DELTA_MAX, alpha=ATTACK_ALPHA,
            scenario=sc, mpc=mpc, with_nominal=False)),
        base_entropy=MC_ENTROPY[KEY],
        n_trials=N_TRIALS_STAGE4,
        proportion_keys=["detected"],
    )
    save_trials_csv(outdir, mc)
    save_metrics(outdir, {
        "experiment": f"{STAGE}/{EXPERIMENT}",
        "research_question": "How stealthy is the attacker against the chi-square detector?",
        "attack": {"attacker_goal": list(ATTACK_GOAL), "attack_start": ATTACK_START,
                   "d_max": ATTACK_D_MAX, "delta_max": ATTACK_DELTA_MAX,
                   "alpha": ATTACK_ALPHA},
        "detection_rule": "persistence: >=4 per-sample alarms within 10 steps (no-attack floor ~0.025)",
        "evaluation": {"type": "monte_carlo", "n_trials": N_TRIALS_STAGE4,
                       "base_entropy": MC_ENTROPY[KEY],
                       "dev_seed_for_figure": DEV_SEEDS[KEY]},
        "metrics": mc.aggregates_to_dict(),
    })
    ar = mc.aggregates["alarm_rate_during_attack"]
    det = mc.aggregates["detected"]
    ttd = mc.aggregates["time_to_detection_steps"]
    print(f"[{STAGE}/{EXPERIMENT}] over {N_TRIALS_STAGE4} trials: "
          f"alarm rate={ar.mean:.3f} (alpha={ATTACK_ALPHA}); "
          f"detection rate={100*det.mean:.0f}% "
          f"(Wilson 95% CI [{100*det.ci95_low:.0f}, {100*det.ci95_high:.0f}]); "
          f"time-to-detect mean={ttd.mean:.1f} steps")


if __name__ == "__main__":
    main()
