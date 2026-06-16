# Design & Derivations

First-principles derivations for every component, written to be reviewed before
an interview. Each section derives the equations the code implements, names the
modeling assumptions, and explains *why* the method is the right one. All four
stages (Kalman filter, MPC, chi-square detector, evasive attacker) are fully
derived and implemented.

Notation: state `x ∈ R^n` (n=4), control `u ∈ R^l` (l=2), measurement
`y ∈ R^m` (m=2). Subscripts `k` index discrete time. `x̂_{k|k-1}` is the prior
(predicted) estimate, `x̂_{k|k}` the posterior (updated) estimate.

---

## 1. System model

The UAV is a 2D double integrator. The continuous-time dynamics per axis are
`p̈ = a` (acceleration is the control). Writing the per-axis state as
`(p, v)` and discretizing with a zero-order hold over a step `dt` (the control is
held constant across the interval):

```
p_{k+1} = p_k + dt·v_k + (dt²/2)·a_k
v_{k+1} = v_k + dt·a_k
```

Stacking both axes gives `x = [px, py, vx, vy]^T`, `u = [ax, ay]^T`, and

```
A = [ I₂  dt·I₂ ]    B = [ (dt²/2)·I₂ ]    C = [ I₂  0₂ ]
    [ 0₂     I₂ ]        [   dt·I₂   ]
```

with GPS observing position only (`C` picks off `px, py`). The stochastic model
adds Gaussian process noise `w_k ~ N(0, Σ_w)` (unmodeled accelerations, wind) and
measurement noise `v_k ~ N(0, Σ_v)` (GPS error):

```
x_{k+1} = A x_k + B u_k + w_k
y_k     = C x_k + d_k + v_k
```

`d_k` is an additive attack on the measurement (GPS spoofing); `d_k = 0` when no
attacker is present. Both noises are zero-mean, white, and mutually independent —
the assumptions under which the Kalman filter is the optimal linear estimator.

---

## 2. Kalman filter (estimation)

### 2.1 Goal

Maintain the posterior belief `p(x_k | y_{1:k})`. Because the dynamics and
measurement are linear and all noise is Gaussian, this posterior is exactly
Gaussian for all `k`; the filter only needs to track its mean `x̂` and
covariance `P`. No approximation is involved — for a linear-Gaussian system the
Kalman filter is the exact Bayesian filter.

### 2.2 Predict step

Given the posterior `x̂_{k-1|k-1}, P_{k-1|k-1}`, push it through the dynamics.
Expectation is linear and `w` is zero-mean:

```
x̂_{k|k-1} = A x̂_{k-1|k-1} + B u_{k-1}
```

For the covariance, with `e = x − x̂` the error propagates as
`e_{k|k-1} = A e_{k-1|k-1} + w_{k-1}`, and since the prior error and `w` are
independent:

```
P_{k|k-1} = A P_{k-1|k-1} Aᵀ + Σ_w
```

Interpretation: propagation *inflates* uncertainty (adds `Σ_w` and stretches by
`A`). Prediction alone makes the estimate progressively less certain.

### 2.3 Update step

A new measurement `y_k` arrives. Define the **innovation** (measurement minus its
prediction) and its covariance:

```
ν_k = y_k − C x̂_{k|k-1}
S_k = C P_{k|k-1} Cᵀ + Σ_v
```

`S_k` is the covariance of `ν_k`: the predicted-measurement uncertainty
`C P_{k|k-1} Cᵀ` plus sensor noise `Σ_v`. The Gaussian conditioning formulas (or
equivalently the linear MMSE estimator) give the posterior:

```
K_k = P_{k|k-1} Cᵀ S_k^{-1}                 (Kalman gain)
x̂_{k|k} = x̂_{k|k-1} + K_k ν_k
P_{k|k} = (I − K_k C) P_{k|k-1}
```

### 2.4 Why the gain is the optimal balance

Choose `K` to minimize the posterior error variance `tr(P_{k|k})`. Writing the
posterior error `e_{k|k} = (I − KC) e_{k|k-1} − K v_k` and expanding (prior error
⊥ sensor noise):

```
P_{k|k}(K) = (I − KC) P_{k|k-1} (I − KC)ᵀ + K Σ_v Kᵀ
```

Setting `∂ tr(P_{k|k}) / ∂K = 0` yields `K = P_{k|k-1} Cᵀ (C P_{k|k-1} Cᵀ +
Σ_v)^{-1} = P_{k|k-1} Cᵀ S^{-1}` — exactly the gain above. The two limits make the
balance concrete:

- **Sensor noisy** (`Σ_v` large ⇒ `S` large): `K → 0`, the update barely moves;
  the filter trusts its model.
- **Prediction uncertain** (`P_{k|k-1}` large): `K` grows toward the pseudo-
  inverse of `C`; the update defers to the measurement.

So `K` is the variance-minimizing compromise between model and sensor, recomputed
every step from their current relative confidence.

### 2.5 Hooks for detection

`ν_k` and `S_k` are surfaced by the implementation (`KFStep`) because the Stage-3
detector is built directly on them (Section 4). This is why the filter is the
foundation of the security layer, not just the navigation layer.

---

## 3. Model Predictive Control

### 3.1 Formulation

At each step, plan a length-`N` trajectory from the current state estimate to the
goal `g`, then apply only the first control (receding horizon):

```
min_{x_{0:N}, u_{0:N-1}}   Σ_{i=1}^{N-1} (x_i − g)ᵀ Q (x_i − g)
                          + (x_N − g)ᵀ Q_f (x_N − g)
                          + Σ_{i=0}^{N-1} u_iᵀ R u_i
s.t.  x_0 = x̂                       (current estimate)
      x_{i+1} = A x_i + B u_i        (dynamics, i = 0..N-1)
      x_lb ≤ x_i ≤ x_ub              (state box, i = 1..N)
      u_lb ≤ u_i ≤ u_ub              (control box, i = 0..N-1)
```

`Q ⪰ 0` penalizes state error, `R ≻ 0` penalizes control effort, `Q_f` is a
terminal weight encouraging arrival. The goal state is `g = [gx, gy, 0, 0]` —
arrive at the goal position *with zero velocity* (the zero-velocity target is
what makes the UAV settle rather than fly through).

### 3.2 Why this is a convex QP (and why that matters)

The cost is a sum of quadratics with PSD weights, hence convex; the dynamics are
affine equalities; the boxes are affine inequalities. A convex quadratic
objective over an affine feasible set is a **quadratic program**, and convexity
means any local optimum is the unique global optimum. There is no initialization
sensitivity and no local-minimum risk — the solver returns *the* optimal plan.
This is the central reason MPC for a linear system with convex costs/constraints
is reliable enough to run in a real-time loop.

### 3.3 Implementation choices and their justifications

- **State box not on `x_0`.** `x_0` is fixed to the measured estimate by the
  equality `x_0 = x̂`. Process noise can place `x̂` slightly outside the box;
  constraining a fixed, infeasible point would make the whole QP infeasible. We
  constrain `x_1..x_N` only.
- **Hard primary, soft fallback.** Hard state constraints are what actually limit
  the *realized* closed-loop velocity: because the plan may not exceed `v_max` at
  any horizon step (including `x_1`), the applied control is forced to brake in
  time. A purely soft bound lets the receding-horizon planner defer braking and
  the realized speed creeps up. But under an unlucky noise draw the estimated
  velocity can exceed `v_max` by more than one step of bounded deceleration can
  remove, making the hard `x_1` box infeasible. So we keep a soft twin (state box
  relaxed by slack `s ≥ 0` with an L1 penalty `ρ·1ᵀs`) and fall back to it only
  when the hard solve is infeasible. The **L1 penalty is exact**: for `ρ` above a
  finite threshold the soft optimum equals the hard optimum whenever the hard
  problem is feasible, so plans still bind exactly in normal conditions. The
  control box is always hard (actuator limits are physical).
- **Cholesky + sum_squares.** We write `(x−g)ᵀ Q (x−g) = ‖L_qᵀ(x−g)‖²` with
  `Q = L_q L_qᵀ`. With parametric `x_init` and `goal`, this keeps the problem
  DPP-compliant so OSQP reuses its matrix factorization across the thousands of
  solves in a Monte Carlo sweep.

### 3.4 Certainty-equivalence and its honest cost

The controller plans on the estimate `x̂` as if it were the true state
(certainty equivalence). Under output feedback with position-only GPS, the KF
velocity estimate lags the true velocity during hard acceleration, so the
realized speed overshoots `v_max` even though every plan respects it. This is a
property of certainty-equivalence output feedback, not a controller defect:
feeding the *true* state to the same controller limits realized speed far more
tightly (the repository measures both). The standard mitigation — constraint
tightening / robust MPC that reserves margin for estimation error — is noted but
not implemented, because the project reports the honest behavior of the
straightforward design rather than tuning for a flattering number.

---

## 4. Chi-square spoofing detector (Stage 3 — implemented)

### 4.1 Statistic

Under the null hypothesis "no attack" (`d_k = 0`), the innovation is zero-mean
Gaussian with covariance `S_k`: `ν_k ~ N(0, S_k)`. Whitening it, the **normalized
innovation squared (NIS)**

```
q_k = ν_kᵀ S_k^{-1} ν_k
```

is a sum of `m` squared independent standard normals, i.e. `q_k ~ χ²(m)` with
`m = 2` here. This is the formal justification for a chi-square test: the
Gaussian noise assumption makes the whitened innovation energy exactly
chi-square-distributed.

### 4.2 Test

Pick a significance level `α` and threshold `τ = χ²_{m, 1−α}` (the `1−α` quantile
of `χ²(m)`). Flag an alarm when `q_k > τ`. Under the null, `P(q_k > τ) = α` by
construction, so **the false-positive rate equals `α`** — provided the filter is
consistent, which Stage 1 Experiment 3 validates empirically (mean NIS ≈ 2,
≈95% of samples inside the 95% band). Under a spoofing attack the innovation mean
shifts away from zero, inflating `E[q_k]` (the statistic becomes noncentral
chi-square), which is what makes the attack detectable.

Stage 3 reports, over many trials: the false-positive rate under no attack
(should match `α`) and the detection rate against a naive large-offset attacker.

### 4.3 Implemented behavior and the absorption effect

Two results from the implementation (N=100 trials each) are worth internalizing:

- **False-positive rate ≈ α.** With no attack the measured per-sample alarm rate
  is 0.050 (95% CI [0.048, 0.053]) against a target `α = 0.05`. The detector's
  calibration is inherited directly from filter consistency (Stage 1): because
  NIS is genuinely `χ²(2)`, thresholding at the `1−α` quantile yields exactly an
  `α` false-alarm rate. No tuning is involved.

- **Naive attacks are caught at onset, then absorbed.** A sudden large constant
  bias (8 m injected offset) is detected in 100% of trials, with a mean
  time-to-detection of 0 steps — the onset produces a huge innovation. But the
  *per-sample* alarm rate across the whole attack window is only ~0.20, because a
  *constant* bias is gradually absorbed by the Kalman filter: the estimate slides
  toward the spoofed measurements until the innovation returns to its normal,
  sub-threshold magnitude. The detector watches for *inconsistency*, and a
  constant offset eventually becomes self-consistent with the filter.

This absorption is the precise opening the Stage 4 attacker exploits: by ramping
the injected offset slowly and keeping the per-sample NIS below `τ` at every
step, an evasive attacker can drag the UAV's estimate without ever producing the
onset spike that betrays the naive attacker.



---

## 5. Evasive attacker (Stage 4 — implemented)

The attacker injects a false measurement offset `d_k` to drive the UAV toward an
attacker-chosen destination while staying below the detector threshold. Because
the offset enters the innovation directly (`ν_k(d_k) = ν_k⁰ + d_k`, with `ν_k⁰`
the no-attack innovation), it shifts both the filter posterior — which steers the
UAV, since the MPC controls on the estimate — and the detector statistic.

### 5.1 The per-step program

At each step `k ≥ k_start` the attacker solves:

```
min_{d}   ‖ b0 + (C K) d − b* ‖²                       (induced-bias objective)
s.t.      (ν0 + d)ᵀ S^{-1} (ν0 + d) ≤ τ                (chi-square stealth)
          ‖d‖ ≤ d_max                                  (magnitude cap)
          ‖d − d_prev‖ ≤ Δ_max                         (ramp-rate cap)
```

### 5.2 Why the objective is an induced-bias target

The closed-loop MPC drives the *estimate* to the legitimate goal `g`. The true
position satisfies `x_true ≈ x̂ − b`, where `b = (estimate − true)` is the bias
the attack induces. At convergence `x̂ ≈ g`, so `x_true ≈ g − b`. To park the
true UAV at the attacker goal `g_att`, the attacker therefore drives the induced
bias toward

```
b* = g − g_att.
```

Writing the posterior position as `C x̂_{k|k} = C x_pred + C K(ν0 + d)`, the
induced bias is `b(d) = b0 + (C K) d` with `b0 = C x_pred + C K ν0 − C x_true`
(the no-attack bias). Minimizing `‖b(d) − b*‖²` is exactly the objective above.

### 5.3 Constraint geometry and solver choice

The stealth constraint `(ν0 + d)ᵀ S^{-1}(ν0 + d) ≤ τ` is an **ellipsoid** in `d`
(since `S^{-1} ≻ 0`) centered at `−ν0`; the magnitude and ramp constraints are
Euclidean balls. The objective is convex quadratic, so the program is in fact a
convex QCQP. We solve it with **scipy SLSQP**, which handles the nonlinear
(quadratic) stealth inequality directly, supplying analytic gradients for the
objective and all three constraints. On solver failure the attacker injects
`d = 0` (the safe stealthy fallback). The implementation is the **strong /
omniscient attacker**: it observes the clean innovation `ν0` (worst case for the
defender). The detector is the *unchanged* Stage 3 per-sample chi-square test.

### 5.4 What the implementation shows (N = 60 trials)

With a genuinely stealthy operating point (`d_max = 1.0`, `Δ_max = 0.2`,
`g_att = (6, 6)` against legit `(10, 10)`):

- **Attack success.** The UAV ends 0.90 m from the legitimate goal (nominal final
  error is ~0.31 m) and 4.79 m from the attacker goal — a real but bounded
  persistent divert.
- **Stealth.** The per-sample alarm rate during the attack is 0.055, essentially
  the `α = 0.05` baseline. Against a calibrated persistence monitor (≥4 alarms in
  10 steps; no-attack floor ~0.025) the attack is detected in only 7% of trials,
  and late (mean ~95 steps in). The gradual ramp constraint is what defeats the
  onset spike that betrayed the naive Stage 3 attacker.
- **Deviation.** Peak deviation from the nominal path is 1.28 m, settling near
  0.90 m.
- **Tradeoff.** Sweeping `d_max` traces a clean Pareto frontier: as `d_max` rises
  0.5 → 2.0, detection climbs 0% → 60% while the final distance to the attacker
  goal falls 5.21 → 4.10 m. More aggressive spoofing is more successful and more
  detectable; the stealth constraint imposes a hard ceiling on how far a
  low-detectability attack can drag the vehicle.

The honest headline: against a well-calibrated chi-square detector, a provably
stealthy attacker's impact is *fundamentally limited* — it can bias the vehicle
by ~1 m while staying at the noise floor, but driving it all the way to the
attacker's goal requires spoofing aggressive enough to be caught.

