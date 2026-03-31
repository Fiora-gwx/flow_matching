## [LRN-20260326-001] correction

**Logged**: 2026-03-26T22:08:00+08:00
**Priority**: high
**Status**: pending
**Area**: backend

### Summary
Do not reuse endpoint-zeroed clock derivatives when converting a base-velocity model back to the true velocity field inside the solver.

### Details
The FT-clock runtime zeroes `ds_dr` at `r=0` and `r=1` to keep endpoint evaluations finite. That is acceptable for analysis helpers, but it is not acceptable for online solver-time velocity conversion. The solver always queries `t=0`, and Heun/RK stages also query `t=1`; if `CFGScaledModel` multiplies the network output by endpoint-zeroed `ds_dr`, the first and last stage velocities collapse to zero for non-uniform clocks, badly degrading sampling quality.

### Suggested Action
When a trained model outputs base velocity and the solver needs the true `dx/dr`, evaluate the clock on a strictly interior-clamped time like `[TIME_EPS, 1 - TIME_EPS]` before multiplying by `ds_dr`.

### Metadata
- Source: user_feedback
- Related Files: examples/image/training/continuous_runtime.py, examples/image/training/eval_loop.py
- Tags: ft-clock, solver, endpoint, velocity-scaling

---

## [LRN-20260328-001] correction

**Logged**: 2026-03-28T15:30:00+08:00
**Priority**: high
**Status**: pending
**Area**: backend

### Summary
For FT-clock base-velocity models, fixing endpoint singularities requires solver-aware terminal stage backoff, not just a tiny global clamp inside `evaluate_clock`.

### Details
The training rewrite to predict base velocity with `q(r) ∝ ds_dr^2` is mathematically fine. The main failure mode appears during sampling, where Heun2 and RK3 query the network at `t=1.0` on the final stage. For low-beta FT clocks, `ds_dr` grows sharply as `r→1`, so replacing `t=1.0` with `1-TIME_EPS` is still far too close to the singular endpoint and can blow up the recovered velocity. The correct fix point is the solver stage time itself: when a model exposes `adapt_solver_time`, the solver should pass the terminal query back inside the last step, e.g. to `1-0.5*dt`, before converting base velocity back to true velocity.

### Suggested Action
Keep training on base velocity, keep importance sampling on `ds_dr^2`, and make fixed-step solvers call a model-provided solver-time adapter so endpoint-sensitive models can move terminal stage queries off `t=1.0`.

### Metadata
- Source: user_feedback
- Related Files: examples/image/training/eval_loop.py, examples/image/training/fixed_step_solver.py
- Tags: ft-clock, solver, endpoint, heun2, rk3
- See Also: LRN-20260326-001

---

## [LRN-20260328-002] correction

**Logged**: 2026-03-28T00:00:00+08:00
**Priority**: high
**Status**: pending
**Area**: tests

### Summary
Evaluation must use a deterministic real-data transform, and metric inputs must keep real/fake preprocessing symmetric.

### Details
The image training pipeline reused the training dataset object for evaluation, so the real benchmark distribution still passed through `RandomHorizontalFlip`. That makes FID, precision/recall, and similar metrics drift across runs because the real reference distribution is no longer fixed. Separately, continuous-model fake samples were quantized to an 8-bit grid before being passed into metric computation, while real samples stayed as float images. That creates an asymmetric preprocessing path and can distort solver/path comparisons. The correct setup is to build a separate evaluation dataset with a deterministic transform and to keep metric inputs as float images in `[0, 1]`, only converting to uint8 when writing image files.

### Suggested Action
Split train and eval transforms, route `data_loader_eval` through the deterministic dataset, and remove pre-metric fake quantization from `eval_loop.py`.

### Metadata
- Source: user_feedback
- Related Files: examples/image/train.py, examples/image/training/data_transform.py, examples/image/training/eval_loop.py
- Tags: evaluation, metrics, fid, precision-recall, preprocessing

---

## [LRN-20260329-001] correction

**Logged**: 2026-03-29T00:00:00+08:00
**Priority**: high
**Status**: pending
**Area**: config

### Summary
When comparing current FT-clock baselines against early git-history runs, check parser defaults first; older solver-sensitivity baselines used different conditioning defaults than the current ablation config.

### Details
The earliest `solver_sensitivity` and `linear_main` configs did not explicitly set `class_drop_prob` or `cfg_scale`, so they inherited parser defaults from the old training code: `class_drop_prob=0.2` and `cfg_scale=0.2`. The newer ablation configs had been set to `class_drop_prob=1.0` and `cfg_scale=0.0`, which silently changed the baseline from a lightly conditioned CFG-trained model into a fully unconditional one. That difference is large enough to dominate FID comparisons and should not be confused with FT-clock endpoint handling or the strict-open-interval time sampler.

### Suggested Action
Pin legacy baseline configs explicitly to `class_drop_prob=0.2` and `cfg_scale=0.2` whenever the goal is an apples-to-apples comparison against the earliest velocity/uniform FT-clock experiments.

### Metadata
- Source: user_feedback
- Related Files: examples/image/train_arg_parser.py, experiments/configs/ft_clock/variance_reduction_ablation.yaml, experiments/configs/ft_clock/variance_reduction_ablation_followup.yaml
- Tags: baseline, config, cfg, conditioning, regression

---

## [LRN-20260328-001] correction

**Logged**: 2026-03-28T00:00:00+08:00
**Priority**: high
**Status**: pending
**Area**: backend

### Summary
For base-velocity FT-clock models, endpoint singularities must be handled at the solver stage-time level, not by a tiny fixed-time clamp inside the model wrapper.

### Details
The u-parameterized training objective is still correct after rewriting the loss against the base velocity and sampling r from q(r) proportional to ds/dr squared. The remaining failure mode is evaluation: Heun2 and RK3 query the model exactly at t=1 on the last corrector stage, and replacing that with a tiny interior clamp such as 1-1e-5 is still too close to the singular endpoint for low-beta FT clocks. That leaves ds/dr numerically huge and blows up the recovered velocity. The correct minimal fix is to let the solver ask the model for an adapted interior stage time based on the last-step width, so the final stage is evaluated strictly inside the last cell instead of at the singular endpoint.

### Suggested Action
Route all fixed-step solver stage times through an optional adapt_solver_time hook and let CFGScaledModel move endpoint queries for base-velocity non-uniform clocks to a step-size-aware interior time.

### Metadata
- Source: user_feedback
- Related Files: examples/image/training/fixed_step_solver.py, examples/image/training/eval_loop.py
- Tags: ft-clock, solver, endpoint, heun2, rk3
- See Also: LRN-20260326-001

---
