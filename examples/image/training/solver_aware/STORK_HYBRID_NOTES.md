# STORK Hybrid Solver-Aware Notes

## Symbols

- `B`: raw NFE budget
- `K_S(B)`: solver-specific effective macro-step count under budget `B`
- terminal-aware defect step:
  `h_eff(s) = min(1 / K_S(B), 1 - s)`

For multi-budget defect aggregation, the normalized curve uses effective step-count scaling:

`Q_tilde_{S,B} = K_S(B)^(2p+2) * Q_{S,B}`

This is not raw-NFE scaling.

## STORK Special Case

STORK is treated as a two-stage solver-aware procedure rather than a single uniform clock:

1. cold-start macro-step:
   fixed Euler startup on `[0, 1 / K_STORK(B)]`
2. warm macro-steps:
   solver-aware allocation only applies on `[1 / K_STORK(B), 1]`

When `K_STORK(B) <= 2`, the warm region is too short to optimize, so node allocation falls back to uniform spacing after preserving the fixed cold-start step.

## Heuristic Boundaries

- Euler / Heun2 exact-budget defect: theorem-backed
- legacy continuous STORK: Heun2-monitor heuristic plus hybrid warm-only node allocation
- defect-based STORK: configured-order warm-state heuristic
- cross-solver transfer: heuristic
- non-exact-budget Heun2 / RK3: heuristic

## STORK Defect Monitor

The STORK defect monitor excludes the cold-start region from optimization.
For `s >= 1 / K_STORK(B)`, it builds a synthetic warm state using:

- `z_prev ~ p_{s_prev}`
- `u_prev = u(z_prev, s_prev)`
- `STORKState(step_index=1, last_velocity=u_prev, last_dt=s - s_prev)`

The resulting warm-state defect is useful as a heuristic monitor, but it is not a strict theorem-backed STORK defect expansion.
