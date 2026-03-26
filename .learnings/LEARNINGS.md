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
