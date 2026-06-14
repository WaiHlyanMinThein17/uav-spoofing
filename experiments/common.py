"""Shared experiment infrastructure.

Centralizes:
  * deterministic seeds (so every experiment is reproducible),
  * the canonical navigation scenario (start/goal, weights, constraints),
  * the results-directory contract: results/<stage>/<experiment>/ holds exactly
    one figure and one metrics.json.

Keeping these here -- rather than copy-pasted across experiment scripts -- is
what makes the runs reproducible and the claims comparable across stages.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# Repository root resolved relative to this file (src-layout aware).
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"

# --- Reproducibility configuration -------------------------------------------
# Two distinct roles, kept deliberately separate:
#
#   DEV_SEEDS       : single fixed seed per experiment, used ONLY for the
#                     one-seed visualization figure (a development run).
#   MC_ENTROPY      : base entropy per experiment from which N independent trial
#                     seeds are spawned for the REPORTED aggregate statistics.
#   N_TRIALS        : number of independent Monte Carlo trials behind every
#                     quantitative claim.
#
# A quantitative number in the README must come from the Monte Carlo path, never
# from the single dev-seed figure.

N_TRIALS = 100

# Single-seed development seeds (figures only).
DEV_SEEDS = {
    "stage1_trajectory": 42,
    "stage1_error": 42,
    "stage1_consistency": 42,
    "stage2_following": 7,
    "stage2_effort": 7,
    "stage2_constraint": 7,
    "stage3_false_positive": 11,
    "stage3_detection": 11,
    "stage4_success": 23,
    "stage4_stealth": 23,
    "stage4_deviation": 23,
    "stage4_tradeoff": 23,
}

# Base entropy for spawning N independent Monte Carlo trial seeds. Distinct from
# DEV_SEEDS so the figure is not silently one of the aggregated trials.
MC_ENTROPY = {
    "stage1_trajectory": 1_000_001,
    "stage1_error": 1_000_002,
    "stage1_consistency": 1_000_003,
    "stage2_following": 2_000_001,
    "stage2_effort": 2_000_002,
    "stage2_constraint": 2_000_003,
    "stage3_false_positive": 3_000_001,
    "stage3_detection": 3_000_002,
    "stage4_success": 4_000_001,
    "stage4_stealth": 4_000_002,
    "stage4_deviation": 4_000_003,
    "stage4_tradeoff": 4_000_004,
}

# Stage 4 trials are ~50x costlier than Stage 1 (the full closed loop is run
# twice per trial plus a per-step SLSQP attacker solve), so the Monte Carlo count
# is reduced for tractability. Confidence intervals reflect this smaller N.
N_TRIALS_STAGE4 = 60

# Canonical evasive-attack configuration used by the Stage 4 experiments. Chosen
# (from a parameter sweep) so the attack is genuinely stealthy: at this setting
# the per-sample alarm rate sits at the alpha=0.05 baseline while the attack
# still drags the UAV off course. The tradeoff study sweeps d_max around this.
ATTACK_GOAL = (6.0, 6.0)      # attacker destination (legit goal is (10, 10))
ATTACK_START = 30             # step at which spoofing begins
ATTACK_D_MAX = 1.0            # max spoof magnitude ||d_k||
ATTACK_DELTA_MAX = 0.2        # max spoof change per step ||d_k - d_{k-1}||
ATTACK_ALPHA = 0.05           # detector significance level (unchanged from Stage 3)

# Backwards-compatible alias (the dev seeds were previously called SEEDS).
SEEDS = DEV_SEEDS


@dataclass(frozen=True)
class NavScenario:
    """Canonical point-to-point navigation scenario shared by control stages."""

    dt: float = 0.1
    horizon: int = 25            # MPC prediction horizon N
    sim_steps: int = 130         # closed-loop simulation length
    start_pos: tuple[float, float] = (0.0, 0.0)
    goal_pos: tuple[float, float] = (10.0, 10.0)

    # Cost weights (diagonal). Position tracking dominates; velocity is mildly
    # penalized so the UAV settles (arrives with ~zero speed); control effort is
    # lightly penalized so the actuator is used but not for free.
    q_pos: float = 10.0
    q_vel: float = 1.0
    qf_scale: float = 12.0       # terminal-cost multiplier on Q
    r_ctrl: float = 0.1

    # Constraints. The speed bound is intentionally set so it BINDS during the
    # aggressive mid-flight phase, making the constraint-satisfaction experiment
    # a meaningful test rather than a vacuous one.
    pos_min: float = -1.0
    pos_max: float = 12.0
    v_max: float = 2.5           # max speed per axis (m/s)
    a_max: float = 3.0           # max commanded acceleration per axis (m/s^2)

    def goal_state(self) -> np.ndarray:
        """Full goal state: arrive at goal position with zero velocity."""
        return np.array([self.goal_pos[0], self.goal_pos[1], 0.0, 0.0])

    def start_state(self) -> np.ndarray:
        return np.array([self.start_pos[0], self.start_pos[1], 0.0, 0.0])

    def Q(self) -> np.ndarray:
        return np.diag([self.q_pos, self.q_pos, self.q_vel, self.q_vel])

    def Qf(self) -> np.ndarray:
        return self.qf_scale * self.Q()

    def R(self) -> np.ndarray:
        return np.diag([self.r_ctrl, self.r_ctrl])

    def state_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lb = np.array([self.pos_min, self.pos_min, -self.v_max, -self.v_max])
        ub = np.array([self.pos_max, self.pos_max, self.v_max, self.v_max])
        return lb, ub

    def control_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lb = np.array([-self.a_max, -self.a_max])
        ub = np.array([self.a_max, self.a_max])
        return lb, ub


def experiment_dir(stage: str, experiment: str) -> Path:
    """Return (and create) results/<stage>/<experiment>/."""
    d = RESULTS_ROOT / stage / experiment
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_metrics(directory: Path, metrics: dict) -> Path:
    """Write metrics.json into an experiment directory."""
    path = Path(directory) / "metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path


def scenario_to_dict(scenario: NavScenario) -> dict:
    """Serialize the scenario config for provenance inside metrics files."""
    return asdict(scenario)
