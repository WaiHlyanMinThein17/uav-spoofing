"""Stage 4 / Experiment 3: Destination deviation.

Research question:
    How far does the attack push the UAV off its nominal trajectory?

Because the nominal and attacked passes use identical pre-drawn noise, the
difference between the two true trajectories is attributable solely to the attack.

Methodology:
    * DEVELOPMENT figure: one seed -> deviation.png (deviation magnitude vs time).
    * REPORTED metrics over N trials: maximum, final, and average deviation
      (averaged over the attack window).

Outputs (results/stage4_attack/destination_deviation/):
    deviation.png, metrics.json, trials.csv
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

STAGE, EXPERIMENT, KEY = "stage4_attack", "destination_deviation", "stage4_deviation"


def _deviation(run: AttackRun) -> np.ndarray:
    return np.linalg.norm(run.true_attacked[:, :2] - run.true_nominal[:, :2], axis=1)


def metrics_from_run(run: AttackRun) -> dict[str, float]:
    dev = _deviation(run)
    win = dev[run.attack_start:]
    return {
        "max_deviation_m": float(dev.max()),
        "final_deviation_m": float(dev[-1]),
        "average_deviation_m": float(win.mean()),
    }


def make_figure(outdir, sc, mpc) -> None:
    run = run_attack(seed=DEV_SEEDS[KEY], attacker_goal=ATTACK_GOAL,
                     attack_start=ATTACK_START, d_max=ATTACK_D_MAX,
                     delta_max=ATTACK_DELTA_MAX, alpha=ATTACK_ALPHA,
                     scenario=sc, mpc=mpc, with_nominal=True)
    dev = _deviation(run)
    fig, ax = new_axes("UAV deviation from nominal trajectory under attack",
                       "time (s)", "deviation (m)")
    ax.plot(run.t, dev, color="C3", label="attacked − nominal")
    ax.axvline(run.t[run.attack_start], color="black", ls=":", lw=1.2,
               label="attack onset")
    ax.legend()
    save_figure(fig, outdir / "deviation.png")


def main() -> None:
    sc = NavScenario()
    outdir = experiment_dir(STAGE, EXPERIMENT)
    mpc = build_controller(LinearUAV(dt=sc.dt), sc)
    make_figure(outdir, sc, mpc)

    mc = run_monte_carlo(
        lambda seed: metrics_from_run(run_attack(
            seed=seed, attacker_goal=ATTACK_GOAL, attack_start=ATTACK_START,
            d_max=ATTACK_D_MAX, delta_max=ATTACK_DELTA_MAX, alpha=ATTACK_ALPHA,
            scenario=sc, mpc=mpc, with_nominal=True)),
        base_entropy=MC_ENTROPY[KEY],
        n_trials=N_TRIALS_STAGE4,
    )
    save_trials_csv(outdir, mc)
    save_metrics(outdir, {
        "experiment": f"{STAGE}/{EXPERIMENT}",
        "research_question": "How far does the attack push the UAV off its nominal path?",
        "attack": {"attacker_goal": list(ATTACK_GOAL), "attack_start": ATTACK_START,
                   "d_max": ATTACK_D_MAX, "delta_max": ATTACK_DELTA_MAX,
                   "alpha": ATTACK_ALPHA},
        "evaluation": {"type": "monte_carlo", "n_trials": N_TRIALS_STAGE4,
                       "base_entropy": MC_ENTROPY[KEY],
                       "dev_seed_for_figure": DEV_SEEDS[KEY]},
        "metrics": mc.aggregates_to_dict(),
    })
    mx, fn, av = (mc.aggregates[k] for k in
                  ("max_deviation_m", "final_deviation_m", "average_deviation_m"))
    print(f"[{STAGE}/{EXPERIMENT}] over {N_TRIALS_STAGE4} trials: "
          f"max dev mean={mx.mean:.3f}±{mx.std:.3f} m; "
          f"final dev mean={fn.mean:.3f} m; avg dev mean={av.mean:.3f} m")


if __name__ == "__main__":
    main()
