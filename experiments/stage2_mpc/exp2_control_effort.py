"""Stage 2 / Experiment 2: Control effort analysis.

Research question:
    What control effort does the MPC expend, and does the applied command stay
    within the actuator limit?

Methodology:
    * DEVELOPMENT figure: one fixed seed -> control_effort.png.
    * REPORTED metrics: total effort, mean accel magnitude, peak accel,
      saturation fraction, max bound overshoot, and the fraction of trials whose
      applied control stays within bounds (0/1 -> Wilson CI), over N trials.

Outputs (results/stage2_mpc/control_effort/):
    control_effort.png, metrics.json, trials.csv
"""

from __future__ import annotations

import numpy as np

from experiments.common import (DEV_SEEDS, MC_ENTROPY, N_TRIALS, NavScenario,
                                experiment_dir, save_metrics)
from experiments.mc import run_monte_carlo, save_trials_csv
from experiments.stage2_mpc._runner import (ClosedLoopRun, build_controller,
                                            run_closed_loop)
from uav_spoof.simulation.dynamics import LinearUAV
from uav_spoof.visualization.plotting import new_axes, save_figure

STAGE, EXPERIMENT, KEY = "stage2_mpc", "control_effort", "stage2_effort"
WITHIN_BOUNDS_TOL = 1e-3  # solver-consistent feasibility tolerance


def metrics_from_run(run: ClosedLoopRun) -> dict[str, float]:
    sc = run.scenario
    u = run.controls
    accel_mag = np.linalg.norm(u, axis=1)
    max_applied = float(np.abs(u).max())
    overshoot = float(max(0.0, max_applied - sc.a_max))
    return {
        "total_control_effort_int_uTu_dt": float(np.sum(accel_mag**2) * sc.dt),
        "mean_accel_magnitude_mps2": float(accel_mag.mean()),
        "peak_accel_mps2": float(np.abs(u).max()),
        "saturation_fraction": float(np.mean(np.isclose(np.abs(u), sc.a_max, atol=1e-3))),
        "max_bound_overshoot_mps2": overshoot,
        "within_bounds": float(overshoot <= WITHIN_BOUNDS_TOL),  # 0/1 outcome
    }


def make_figure(outdir, sc: NavScenario, mpc) -> None:
    run = run_closed_loop(seed=DEV_SEEDS[KEY], scenario=sc, mpc=mpc)  # dev seed
    u = run.controls
    fig, ax = new_axes("MPC applied control effort (no attack)",
                       "time (s)", r"commanded acceleration (m/s$^2$)")
    ax.plot(run.t, u[:, 0], color="C0", label=r"$a_x$")
    ax.plot(run.t, u[:, 1], color="C4", label=r"$a_y$")
    ax.axhline(sc.a_max, color="black", ls="--", lw=1, label=r"$\pm a_{max}$")
    ax.axhline(-sc.a_max, color="black", ls="--", lw=1)
    ax.legend()
    save_figure(fig, outdir / "control_effort.png")


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
        proportion_keys=["within_bounds"],
    )
    save_trials_csv(outdir, mc)
    save_metrics(outdir, {
        "experiment": f"{STAGE}/{EXPERIMENT}",
        "research_question": "What control effort is used and is it within a_max?",
        "a_max": sc.a_max, "within_bounds_tol": WITHIN_BOUNDS_TOL,
        "evaluation": {"type": "monte_carlo", "n_trials": N_TRIALS,
                       "base_entropy": MC_ENTROPY[KEY],
                       "dev_seed_for_figure": DEV_SEEDS[KEY]},
        "metrics": mc.aggregates_to_dict(),
    })
    eff = mc.aggregates["total_control_effort_int_uTu_dt"]
    wb = mc.aggregates["within_bounds"]
    print(f"[{STAGE}/{EXPERIMENT}] over {N_TRIALS} trials: "
          f"total effort mean={eff.mean:.3f}±{eff.std:.3f}; "
          f"within bounds in {100*wb.mean:.0f}% of trials "
          f"(Wilson 95% CI [{100*wb.ci95_low:.0f}, {100*wb.ci95_high:.0f}])")


if __name__ == "__main__":
    main()
