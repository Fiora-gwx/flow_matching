# Defect-Based Solver-Aware Clock on Path Distributions

## 1. Positioning

这条分支是**新增**在现有 framework 之上的：

- old theory kept:
  旧的 `FT-clock / finite-time stability / legacy continuous solver-aware monitor` 全部保留
- new branch added:
  新增 `defect_based` monitor family

默认行为仍然不变：

- `solver_aware_clock_mode=off`
- `solver_aware_monitor_family=legacy_continuous`

所以不传新参数时，旧逻辑完全不变。

## 2. Why Path-Based Defect

defect-based monitor 的目标是做：

- solver-specific
- budget-specific
- training-free

但它不应该依赖模型自己 rollout 出来的经验轨迹分布。原因很直接：

- 对每个样本都 rollout 一条模型轨迹，成本本身就很高
- 如果再把这些 rollout 当成 defect monitor 的主期望分布，训练前和评估前的额外成本会过大
- 这也会把 monitor 定义绑到一个额外的 model-generated distribution 上

对于本仓库已经已知的概率路径，最自然的做法是直接在路径分布 `p_s` 上定义 defect monitor。

以 linear path 为例：

`Z_s = (1-s) epsilon + s x,  x ~ p_data, epsilon ~ p_0`

因此给定任意 `s`，我们都可以直接采样 `z ~ p_s`。

## 3. General Theorem

考虑边缘 ODE：

`dx / ds = u(x, s),  s in [0, 1]`

对任意 solver `S`，记一步映射为：

`Psi_h^S(z, s)`

定义 self-consistency defect：

`Delta_S(z, s; h) = Psi_h^S(z, s) - Psi_{h/2}^{S,(2)}(z, s)`

若 solver `S` 是 `p` 阶，并满足局部误差展开：

`Psi_h^S(z, s) = Phi_{s+h,s}(z) + C_S h^{p+1} E_S[u](z, s) + O(h^{p+2})`

则有：

`Delta_S(z, s; h) = C_S (1 - 2^{-p}) h^{p+1} E_S[u](z, s) + O(h^{p+2})`

所以 defect 和原来的连续局部误差 proxy 在小步极限下是一致的，只是这里直接用 solver 自己的一步自一致误差来做代理。

## 4. Path-Distribution-Based Monitor

把期望对象定义在 `p_s` 上：

`Q_{S,N}^{path}(s) = E_{z ~ p_s} || Delta_S(z, s; h_N) ||^2`

这里的实现含义是：

- `z ~ p_s`
- solver-specific
- budget-specific
- training-free

对 multi-stage solver，本仓库实现里先把用户给的 NFE budget 映射到真实执行的 macro-step count `K_S(N)`，再使用：

`h_N = 1 / K_S(N)`

原因是 clock 需要和采样时真正执行的 solver macro-step 对齐。

## 5. Optimal Density

令：

`M_{S,N}(s) = sqrt(Q_{S,N}^{path}(s))`

则固定 budget 下仍然有最优密度：

`rho_{S,N}^*(s) propto M_{S,N}(s)^{1/(p+1)}`

等价地：

`rho_{S,N}^*(s) propto (Q_{S,N}^{path}(s) + eps)^{1 / [2(p+1)]}`

所以 defect-based 分支和旧 solver-aware 分支在 clock construction 形式上完全兼容，只是 monitor 从连续导数型 `Q` 改成了 path-based defect `Q`。

## 6. Solver-Specific Formulas

### Euler

`Psi_h^E(z, s) = z + h u(z, s)`

`Psi_{h/2}^{E,(2)}(z, s) = z + h/2 u(z, s) + h/2 u(z + h/2 u(z, s), s + h/2)`

`Q_{E,N}^{path}(s) = E_{z ~ p_s} || Psi_h^E(z,s) - Psi_{h/2}^{E,(2)}(z,s) ||^2`

`rho_{E,N}(s) propto (Q_{E,N}^{path}(s) + eps)^(1/4)`

### Heun2

`k1 = u(z, s)`

`k2 = u(z + h k1, s + h)`

`Psi_h^H(z, s) = z + h/2 (k1 + k2)`

`Q_{H,N}^{path}(s) = E_{z ~ p_s} || Psi_h^H(z,s) - Psi_{h/2}^{H,(2)}(z,s) ||^2`

`rho_{H,N}(s) propto (Q_{H,N}^{path}(s) + eps)^(1/6)`

### STORK

把 STORK macro-step 记为：

`Psi_h^{STORK}(z, s)`

则：

`Q_{STORK,N}^{path}(s) = E_{z ~ p_s} || Psi_h^{STORK}(z,s) - Psi_{h/2}^{STORK,(2)}(z,s) ||^2`

当前实现里不强行声称 STORK 一定有严格 theorem-backed 的固定阶数，而是使用：

- configured effective order `p_stork`

并明确记录：

- assumed / configured effective order

于是：

`rho_{STORK,N}(s) propto (Q_{STORK,N}^{path}(s) + eps)^{1 / [2(p_stork+1)]}`

## 7. Multi-Budget Aggregation

如果每个 budget 都各自造一套独立 clock，后续会很乱，所以对预算集合 `N_set` 使用归一化聚合：

`Q_tilde_{S,N}^{path}(s) = N^{2p+2} Q_{S,N}^{path}(s)`

再定义：

`M_tilde_{S,N_set}(s) = sum_N w_N sqrt(Q_tilde_{S,N}^{path}(s))`

其中：

- `w_N > 0`
- `sum_N w_N = 1`

最终共享密度：

`rho_{S,N_set}(s) propto M_tilde_{S,N_set}(s)^{1/(p+1)}`

为什么需要先做 `N^{2p+2}` 归一化：

- defect 主量级是 `h^{p+1}`
- 平方范数之后变成 `h^{2p+2}`
- 不先归一化，不同 budgets 之间只会被纯量级差异主导

## 8. Implementation Scope

phase-1 当前只实现：

- `k=0`
- training-free
- path-distribution-based defect monitor
- single-budget and multi-budget
- eval-only

没有使用：

- model-generated rollout distribution
- `mu_s^ref`
- reference trajectory cache

## 9. Main Files

- `examples/image/training/solver_aware/defect_monitor.py`
- `examples/image/training/solver_aware/defect_clock.py`
- `examples/image/training/solver_aware/fixed_point_defect.py`
- `examples/image/training/solver_aware/fixed_point.py`

旧分支仍保留：

- `examples/image/training/solver_aware/monitors.py`
- `examples/image/training/solver_aware/clock.py`

所以现在仓库中存在两条并行 solver-aware 路线：

1. `legacy_continuous`
2. `defect_based`
