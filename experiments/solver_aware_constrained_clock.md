# Solver-Aware Constrained Clock

## Why The Main Method Changed

旧的 unconstrained solver-aware / propagation-aware 把连续密度直接写成

- solver-aware: `rho(s) ∝ M(s)^(1/(p+1))`
- propagation-aware: `rho(s) ∝ G(s)^(1/(p+1)) M(s)^(1/(p+1))`

这在低 NFE 下会系统性地产生 node collapse：某些区间被分到过少节点，导致单步步长过大，局部误差展开失效。Euler 上已经观测到 solver-aware 劣于 uniform baseline，propagation-aware 在低 NFE 下还会出现严重尾部大步长，因此主方法不能继续保持 unconstrained 版本。

从现在开始，仓库里的 solver-aware / propagation-aware 主路径统一切换为 constrained formulation。旧 unconstrained 路径只保留为 deprecated debug mode，通过 `--solver_aware_legacy_unconstrained` 显式开启。

## Core Theory

考虑边缘 ODE

`dx/ds = u(x, s), s in [0,1]`

定义物质导数

`L_u := ∂_s + u · ∇_x`

并记

- `A(s) = sup_x ||L_u u(x,s)||`
- `B(s) = sup_x ||L_u^2 u(x,s)||`
- `G(s) = exp(∫_s^1 ell(t) dt)`

其中 `ell(s)` 满足 `||J_x u(x,s)|| <= ell(s)`。

### Euler admissible floor

Euler 一步局部缺陷满足

`||d(s,h)|| <= (h^2 / 2) A(s) + (h^3 / 6) B(s) + O(h^4)`

给定 `eta in (0,1)`，要求三阶余项不超过二阶主项的 `eta` 倍：

`(h^3 / 6) B(s) <= eta * (h^2 / 2) A(s)`

得到 admissible step

`h_adm(s) = 3 eta A(s) / B(s)`

若 `h(s) ≈ 1 / (N rho(s))`，则必须满足

`rho(s) >= rho_floor_N(s) = B(s) / (3 eta N A(s))`

### Constrained objective

全局传播感知代理写成

`J_N[rho] = ∫ a(s) / rho(s) ds`

其中 `a(s) = G(s) A(s)`。正确问题不是无约束最小化，而是

`min_rho ∫ a(s)/rho(s) ds`

subject to

- `∫ rho(s) ds = 1`
- `rho(s) >= rho_floor_N(s)`

其唯一极小解为

`rho_N*(s) = max{ rho_floor_N(s), c_N sqrt(a(s)) }`

其中 `c_N` 由归一化唯一确定：

`∫ max{ rho_floor_N(s), c_N sqrt(a(s)) } ds = 1`

## Implemented Proxy Version

代码里使用 training-free monitor 代理：

- `A(s) ≈ sqrt(Q_E(s) + eps)`
- `B(s) ≈ sqrt(Q_H(s) + eps)`

其中

- `Q_E(s) ≈ E ||L_u u||^2`
- `Q_H(s) ≈ E ||L_u^2 u||^2`

因此

`rho_floor_N(s) ≈ (1 / (3 eta N)) * sqrt((Q_H(s)+eps)/(Q_E(s)+eps))`

### Constrained solver-aware Euler

无传播项时：

`rho_E_constrained(s) = max{ rho_floor_N(s), c_N (Q_E(s)+eps)^(1/4) }`

### Constrained propagation-aware Euler

有传播项时：

`rho_E_constrained(s) = max{ rho_floor_N(s), c_N G(s)^(1/2) (Q_E(s)+eps)^(1/4) }`

## Solver Status

- Euler: constrained theorem-backed proxy
- Heun2: constrained proxy extension
- STORK4: constrained proxy extension using Heun2-style monitor construction

这里的 “proxy” 指 `Q_E / Q_H / G` 仍然来自有限样本与数值估计；但主优化问题与可行域已经切换到 constrained formulation。

## Code Path

- `examples/image/training/solver_aware/monitors.py`
  负责估计 `Q_E(s)` 与 `Q_H(s)`
- `examples/image/training/solver_aware/propagation.py`
  负责估计 `ell(s)` 与 `G(s)`
- `examples/image/training/solver_aware/clock.py`
  负责构造 `rho_floor`、求解 `rho = max{floor, c w}`、生成 `phi` 与 nodes
- `examples/image/training/solver_aware/fixed_point.py`
  负责 shared profile 缓存、per-NFE constrained artifacts、以及 fixed-point finetune 入口

## Fixed-Point Interpretation

- `k=0`: training-free constrained solver-aware / propagation-aware
- `k>=1`: constrained fixed-point finetuning

每一轮：

1. 从上一轮 checkpoint 出发
2. 估计 `Q_E / Q_H / optional G`
3. 构造 constrained density
4. 生成新 clock / nodes
5. continuation finetune `solver_aware_finetune_epochs`
6. 保存 `iter_k` checkpoint
7. 评估

## Important Output Fields

artifact/profile 会保存并导出：

- `q_values`
- `q_smoothed`
- `q_h_values`
- `q_h_smoothed`
- `rho_floor`
- `unconstrained_weight`
- `final_density`
- `phi`
- `g_values`
- `nodes`
- `step_sizes`

旧 unconstrained 路径不再是主实验配置，也不再作为主表结论。
