#!/usr/bin/env bash
# Reproduce every test, figure, and metric in the repository from scratch.
# Usage:  bash reproduce_all.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "=== 1/3  Tests ==="
python -m pytest -q

echo "=== 2/3  Clearing previous results ==="
rm -rf results

echo "=== 3/3  Running all experiments (Stages 1-4) ==="

echo "--- Stage 1: Kalman filter ---"
python -m experiments.stage1_kalman.exp1_trajectory_tracking
python -m experiments.stage1_kalman.exp2_error_analysis
python -m experiments.stage1_kalman.exp3_consistency

echo "--- Stage 2: MPC ---"
python -m experiments.stage2_mpc.exp1_trajectory_following
python -m experiments.stage2_mpc.exp2_control_effort
python -m experiments.stage2_mpc.exp3_constraint_satisfaction

echo "--- Stage 3: Chi-square detector ---"
python -m experiments.stage3_detection.exp1_false_positive_rate
python -m experiments.stage3_detection.exp2_detection_rate

echo "--- Stage 4: Evasive attacker ---"
python -m experiments.stage4_attack.exp1_attack_success
python -m experiments.stage4_attack.exp2_stealthiness
python -m experiments.stage4_attack.exp3_destination_deviation
python -m experiments.stage4_attack.exp4_tradeoff_study

echo "=== Done. Figures, metrics.json, and trials.csv are under results/ ==="
