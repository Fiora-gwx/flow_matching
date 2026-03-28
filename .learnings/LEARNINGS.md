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
