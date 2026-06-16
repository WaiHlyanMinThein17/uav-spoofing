# Setup & Reproduction Guide

This guide covers environment setup, pushing the repo to GitHub, configuring
VS Code, and reproducing every figure and metric.

---

## 1. Environment

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev]"              # installs the package + pytest
```

Runtime dependencies (numpy, scipy, cvxpy, matplotlib) are declared in
`pyproject.toml` / `requirements.txt`. The MPC solves a convex QP through
cvxpy + OSQP; the evasive attacker uses scipy SLSQP.

Verify the install:

```bash
python -m pytest -q          # expect: 23 passed
```

---

## 2. Push to GitHub (private repo)

Create an empty **private** repository on GitHub (no README/.gitignore — this repo
already has them), then:

```bash
git init
git add .
git commit -m "UAV GPS-spoofing attacker-defender: Stages 1-4 complete"
git branch -M main
git remote add origin git@github.com:<your-username>/uav-spoofing.git
git push -u origin main
```

Using HTTPS instead of SSH:

```bash
git remote add origin https://github.com/<your-username>/uav-spoofing.git
git push -u origin main
```

`results/` is committed so the figures and metrics are viewable on GitHub without
re-running. To keep generated artifacts out of version control instead, add
`results/` to `.gitignore` before the first commit and regenerate locally with
`bash reproduce_all.sh`.

---

## 3. VS Code

Recommended extensions: **Python** and **Pylance** (Microsoft).

1. Open the folder: `code .`
2. Select the interpreter: `Ctrl/Cmd+Shift+P` -> "Python: Select Interpreter" ->
   choose `./.venv`.
3. Tests: open the Testing panel (beaker icon); pytest is auto-discovered from
   `pyproject.toml`. Or run `python -m pytest` in the integrated terminal.

A minimal `.vscode/settings.json` (optional):

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"]
}
```

Run a single experiment from the integrated terminal (module form, from the repo
root, so package imports resolve):

```bash
python -m experiments.stage4_attack.exp1_attack_success
```

---

## 4. Reproduce every figure and metric

One command runs the tests, clears `results/`, and regenerates everything:

```bash
bash reproduce_all.sh
```

Or run experiments individually (each writes only its own output directory):

```bash
# Stage 1 — Kalman filter
python -m experiments.stage1_kalman.exp1_trajectory_tracking
python -m experiments.stage1_kalman.exp2_error_analysis
python -m experiments.stage1_kalman.exp3_consistency

# Stage 2 — MPC
python -m experiments.stage2_mpc.exp1_trajectory_following
python -m experiments.stage2_mpc.exp2_control_effort
python -m experiments.stage2_mpc.exp3_constraint_satisfaction

# Stage 3 — Chi-square detector
python -m experiments.stage3_detection.exp1_false_positive_rate
python -m experiments.stage3_detection.exp2_detection_rate

# Stage 4 — Evasive attacker
python -m experiments.stage4_attack.exp1_attack_success
python -m experiments.stage4_attack.exp2_stealthiness
python -m experiments.stage4_attack.exp3_destination_deviation
python -m experiments.stage4_attack.exp4_tradeoff_study
```

Each experiment writes to `results/<stage>/<experiment>/`:

| File | Contents |
|---|---|
| `*.png` | the single-claim figure (one development seed) |
| `metrics.json` | aggregated Monte Carlo statistics (mean/std/min/max/CI) |
| `trials.csv` | raw per-trial values (so any aggregate can be re-derived) |

All seeds are fixed in `experiments/common.py`, so every run is deterministic and
the reported numbers reproduce exactly.

### Runtime notes

- Stage 1 and Stage 3 experiments are fast (seconds).
- Stage 2 experiments run 100 closed-loop trials each (~1–2 minutes per script).
- Stage 4 experiments run the full closed loop (plus a per-step SLSQP attacker
  solve), so they use N = 60 trials; each script takes roughly 1–2 minutes. The
  tradeoff study sweeps 6 values of `d_max`.
