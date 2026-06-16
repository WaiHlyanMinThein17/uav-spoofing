"""Stage 4 / Experiment 4: Attack tradeoff study.

Research question:
    How does the attacker's success trade off against its probability of being
    detected as the maximum spoofing magnitude d_max is varied?

Method:
    Sweep d_max (ramp-rate delta_max fixed). For each value, run N paired trials
    (identical seeds across the sweep, so points differ only in d_max) and
    measure attack success and detection probability.

Success axis: final distance to the attacker goal (lower = more successful).
Detection axis: per-trial persistence-rule detection probability.

Methodology note: aggregates use the shared Monte Carlo helpers (Student-t for
continuous, Wilson for the detection proportion). trials.csv records every
(d_max, seed) row.

Outputs (results/stage4_attack/tradeoff_study/):
    tradeoff.png, metrics.json, trials.csv
"""

from __future__ import annotations

import csv

import numpy as np

from experiments.common import (ATTACK_ALPHA, ATTACK_DELTA_MAX, ATTACK_GOAL,
                                ATTACK_START, MC_ENTROPY, NavScenario,
                                experiment_dir, save_metrics)
from experiments.mc import aggregate, make_seeds
from experiments.stage4_attack._runner import (build_controller,
                                               persistence_detection, run_attack)
from uav_spoof.simulation.dynamics import LinearUAV
from uav_spoof.visualization.plotting import new_axes, save_figure

STAGE, EXPERIMENT, KEY = "stage4_attack", "tradeoff_study", "stage4_tradeoff"
D_MAX_SWEEP = [0.5, 0.8, 1.1, 1.4, 1.7, 2.0]
TRADEOFF_TRIALS = 15


def _trial(seed: int, d_max: float, sc, mpc) -> dict[str, float]:
    run = run_attack(seed=seed, attacker_goal=ATTACK_GOAL, attack_start=ATTACK_START,
                     d_max=d_max, delta_max=ATTACK_DELTA_MAX, alpha=ATTACK_ALPHA,
                     scenario=sc, mpc=mpc, with_nominal=False)
    final_pos = run.true_attacked[-1, :2]
    detected = persistence_detection(run.alarm, run.attack_start) != -1
    return {
        "final_distance_to_attacker_goal_m": float(np.linalg.norm(final_pos - run.attacker_goal)),
        "final_distance_to_legit_goal_m": float(np.linalg.norm(final_pos - run.legit_goal[:2])),
        "alarm_rate_during_attack": float(run.alarm[run.attack_start:].mean()),
        "detected": float(detected),
    }


def main() -> None:
    sc = NavScenario()
    outdir = experiment_dir(STAGE, EXPERIMENT)
    mpc = build_controller(LinearUAV(dt=sc.dt), sc)
    seeds = make_seeds(MC_ENTROPY[KEY], TRADEOFF_TRIALS)  # shared across d_max

    table = []
    all_rows = []
    for d_max in D_MAX_SWEEP:
        trials = [_trial(s, d_max, sc, mpc) for s in seeds]
        for s, tr in zip(seeds, trials):
            all_rows.append({"d_max": d_max, "seed": s, **tr})
        keys = trials[0].keys()
        agg = {k: aggregate([t[k] for t in trials],
                            proportion=(k == "detected")) for k in keys}
        table.append({
            "d_max": d_max,
            "detection_probability": agg["detected"].mean,
            "detection_ci95_low": agg["detected"].ci95_low,
            "detection_ci95_high": agg["detected"].ci95_high,
            "final_dist_attacker_goal_mean": agg["final_distance_to_attacker_goal_m"].mean,
            "final_dist_attacker_goal_std": agg["final_distance_to_attacker_goal_m"].std,
            "final_dist_legit_goal_mean": agg["final_distance_to_legit_goal_m"].mean,
            "alarm_rate_mean": agg["alarm_rate_during_attack"].mean,
        })

    # --- figure: success (final dist to attacker goal, lower better) vs detection ---
    det = [row["detection_probability"] for row in table]
    succ = [row["final_dist_attacker_goal_mean"] for row in table]
    fig, ax = new_axes("Attack tradeoff: success vs detection probability",
                       "detection probability", "final dist to attacker goal (m)")
    ax.plot(det, succ, "-o", color="C3")
    for row in table:
        ax.annotate(f"d={row['d_max']:.1f}",
                    (row["detection_probability"], row["final_dist_attacker_goal_mean"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    save_figure(fig, outdir / "tradeoff.png")

    # --- per-trial CSV (every d_max x seed row) ---
    csv_path = outdir / "trials.csv"
    cols = ["d_max", "seed", "final_distance_to_attacker_goal_m",
            "final_distance_to_legit_goal_m", "alarm_rate_during_attack", "detected"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)

    save_metrics(outdir, {
        "experiment": f"{STAGE}/{EXPERIMENT}",
        "research_question": "How does attack success trade off against detection?",
        "swept_parameter": "d_max",
        "fixed": {"delta_max": ATTACK_DELTA_MAX, "attacker_goal": list(ATTACK_GOAL),
                  "attack_start": ATTACK_START, "alpha": ATTACK_ALPHA},
        "evaluation": {"type": "monte_carlo_sweep", "trials_per_point": TRADEOFF_TRIALS,
                       "base_entropy": MC_ENTROPY[KEY], "d_max_values": D_MAX_SWEEP},
        "detection_rule": "persistence: >=4 per-sample alarms within 10 steps (no-attack floor ~0.025)",
        "pareto_table": table,
    })

    print(f"[{STAGE}/{EXPERIMENT}] sweep over d_max={D_MAX_SWEEP}, "
          f"{TRADEOFF_TRIALS} trials each:")
    for row in table:
        print(f"   d_max={row['d_max']:.1f}: detection={row['detection_probability']:.2f} "
              f"final_dist_attacker={row['final_dist_attacker_goal_mean']:.2f} m "
              f"alarm_rate={row['alarm_rate_mean']:.3f}")


if __name__ == "__main__":
    main()
