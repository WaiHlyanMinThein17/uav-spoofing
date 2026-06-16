"""Stage 4 / Experiment 1: Attack success.

Research question:
    Does the evasive attacker redirect the UAV toward the attacker's destination
    and away from the legitimate goal?

Methodology:
    * DEVELOPMENT figure: one seed -> trajectory.png (nominal vs attacked path,
      with legit and attacker goals).
    * REPORTED metrics: final distance to attacker goal and to legitimate goal,
      aggregated over N independent trials.

Outputs (results/stage4_attack/attack_success/):
    trajectory.png, metrics.json, trials.csv
"""

from __future__ import annotations

import numpy as np

from experiments.common import (ATTACK_ALPHA, ATTACK_D_MAX, ATTACK_DELTA_MAX,
                                ATTACK_GOAL, ATTACK_START, DEV_SEEDS, MC_ENTROPY,
                                N_TRIALS_STAGE4, NavScenario, experiment_dir,
                                save_metrics)
from experiments.mc import run_monte_carlo, save_trials_csv
from experiments.stage4_attack._runner import (AttackRun, build_controller,
                                               run_attack)
from uav_spoof.simulation.dynamics import LinearUAV
from uav_spoof.visualization.plotting import new_axes, save_figure

STAGE, EXPERIMENT, KEY = "stage4_attack", "attack_success", "stage4_success"


def metrics_from_run(run: AttackRun) -> dict[str, float]:
    final_pos = run.true_attacked[-1, :2]
    return {
        "final_distance_to_attacker_goal_m": float(np.linalg.norm(final_pos - run.attacker_goal)),
        "final_distance_to_legit_goal_m": float(np.linalg.norm(final_pos - run.legit_goal[:2])),
    }


def make_figure(outdir, sc, mpc) -> None:
    run = run_attack(seed=DEV_SEEDS[KEY], attacker_goal=ATTACK_GOAL,
                     attack_start=ATTACK_START, d_max=ATTACK_D_MAX,
                     delta_max=ATTACK_DELTA_MAX, alpha=ATTACK_ALPHA,
                     scenario=sc, mpc=mpc, with_nominal=True)
    fig, ax = new_axes("UAV trajectory under evasive attack",
                       "x (m)", "y (m)")
    ax.plot(run.true_nominal[:, 0], run.true_nominal[:, 1], color="C0",
            label="nominal (no attack)")
    ax.plot(run.true_attacked[:, 0], run.true_attacked[:, 1], color="C3",
            label="attacked")
    ax.scatter(*sc.start_pos, color="C2", s=70, marker="o", label="start", zorder=5)
    ax.scatter(*sc.goal_pos, color="C0", s=130, marker="*",
               label="legit goal", zorder=5)
    ax.scatter(*ATTACK_GOAL, color="C3", s=130, marker="X",
               label="attacker goal", zorder=5)
    ax.axis("equal")
    ax.legend()
    save_figure(fig, outdir / "trajectory.png")


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
    )
    save_trials_csv(outdir, mc)
    save_metrics(outdir, {
        "experiment": f"{STAGE}/{EXPERIMENT}",
        "research_question": "Does the attacker redirect the UAV toward its goal?",
        "attack": {"attacker_goal": list(ATTACK_GOAL), "legit_goal": list(sc.goal_pos),
                   "attack_start": ATTACK_START, "d_max": ATTACK_D_MAX,
                   "delta_max": ATTACK_DELTA_MAX, "alpha": ATTACK_ALPHA},
        "evaluation": {"type": "monte_carlo", "n_trials": N_TRIALS_STAGE4,
                       "base_entropy": MC_ENTROPY[KEY],
                       "dev_seed_for_figure": DEV_SEEDS[KEY]},
        "metrics": mc.aggregates_to_dict(),
    })
    ga = mc.aggregates["final_distance_to_attacker_goal_m"]
    gl = mc.aggregates["final_distance_to_legit_goal_m"]
    print(f"[{STAGE}/{EXPERIMENT}] over {N_TRIALS_STAGE4} trials: "
          f"final dist to attacker goal mean={ga.mean:.3f}±{ga.std:.3f} m; "
          f"final dist to legit goal mean={gl.mean:.3f}±{gl.std:.3f} m")


if __name__ == "__main__":
    main()
