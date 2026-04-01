# Solver-Aware Clock Phase-1

> Deprecated note:
> This document records the old unconstrained phase-1 solver-aware path.
> The repository main path now uses the constrained formulation documented in
> `experiments/solver_aware_constrained_clock.md`. The unconstrained path only
> remains as a deprecated debug mode.

## 1. 原理

我们在现有 FT-Clock / finite-time contraction 主线之外，新增一条并行的 solver-aware clock 支路。当前阶段只实现 `k=0` 的 training-free 版本：

- 固定一个已知 sampling solver `S`
- 使用已经训练好的边缘场 `u(x, s)`
- 估计 solver-specific monitor `Q_S(s)`
- 构造新的累计时钟 `phi(s)` 与反函数 `psi(r)`
- 生成非均匀节点 `s_n = psi(n / N)`
- 在不重训模型的前提下，直接用这些 nodes 做 few-step sampling

这里的边缘 ODE 为：

`dx / ds = u(x, s), s in [0, 1]`

定义物质导数：

`L_u := partial_s + u · grad_x`

因此：

- `L_u u = partial_s u + J_x u · u`
- `L_u^2 u = L_u(L_u u)`

## 2. 公式

### Euler

Euler 的局部截断误差主项由 `L_u u` 控制，因此定义：

`Q_E(s) = E_{z ~ p_s} ||L_u u(z, s)||^2`

对应的 phase-1 最优误差代理密度为：

`rho_E(s) = dr / ds propto (Q_E(s) + eps)^(1/4)`

累计时钟：

`phi_E(s) = int_0^s (Q_E(u) + eps)^(1/4) du / int_0^1 (Q_E(u) + eps)^(1/4) du`

### Heun2

Heun2 的局部截断误差主项由 `L_u^2 u` 控制，因此定义：

`Q_H(s) = E_{z ~ p_s} ||L_u^2 u(z, s)||^2`

对应的 phase-1 最优误差代理密度为：

`rho_H(s) = dr / ds propto (Q_H(s) + eps)^(1/6)`

累计时钟：

`phi_H(s) = int_0^s (Q_H(u) + eps)^(1/6) du / int_0^1 (Q_H(u) + eps)^(1/6) du`

### STORK4

当前版本不声称已经完成严格的 STORK4 最优 monitor 定理。phase-1 只完成：

- STORK4 在 arbitrary non-uniform nodes 上运行
- super-step 局部步长 `h_n = s_{n+1} - s_n`
- stage 时间 `tau_{n,j} = s_n + c_j h_n`
- actual NFE 与 virtual NFE 分离计数
- flow matching 场景下先使用 first-order Taylor virtual NFE

当 `solver_aware_target_solver=stork4` 时，当前实现会显式记录：monitor 仍复用 Heun2 proxy，属于 heuristic phase-1，而不是 theorem-backed 结论。

## 3. 新参数

- `--solver_aware_clock_mode {off,training_free,fixed_point}`
- `--solver_aware_target_solver {euler,heun2,stork4}`
- `--solver_aware_k`
- `--solver_aware_monitor_estimator {auto,jvp,fd}`
- `--solver_aware_monitor_grid_size`
- `--solver_aware_monitor_batch_size`
- `--solver_aware_eps`
- `--solver_aware_cache_path`
- `--solver_aware_use_nodes`
- `--solver_aware_checkpoint_path`
- `--solver_aware_checkpoint_from_experiment`
- `--solver_aware_checkpoint_epoch`

默认值保持旧行为不变：`solver_aware_clock_mode=off` 且 `solver_aware_use_nodes=false` 时，旧 FT-Clock / uniform / solver 逻辑完全不变。

## 4. 数据流 / 代码流程

1. `examples/image/train_arg_parser.py`
   新增 solver-aware CLI 参数。
2. `examples/image/train.py`
   在 `eval_only + solver-aware` 情况下解析 checkpoint 来源。
3. `examples/image/training/solver_aware/monitors.py`
   估计 Euler / Heun2 monitor。
4. `examples/image/training/solver_aware/clock.py`
   从 `Q(s)` 构造 `phi(s)`、`psi(r)` 和 non-uniform nodes。
5. `examples/image/training/solver_aware/fixed_point.py`
   封装 phase-1 的 `k=0 training_free` 路径，并预留未来 `k>=1` fixed-point 接口。
6. `examples/image/training/fixed_step_solver.py`
   统一支持 arbitrary time grid，供 Euler / Heun2 / STORK4 复用。
7. `experiments/run_experiments.py`
   把新参数透传到评估命令，并把 solver-aware 元数据写入 CSV。
8. `experiments/visualize_solver_aware_results.py`
   汇总 FID 曲线、monitor / phi / nodes 图和 summary markdown。

## 5. 实验设置

phase-1 配置文件：

- `experiments/configs/ft_clock/solver_aware_training_free_linear_uniform.yaml`

当前实验固定：

- checkpoint: `linear + uniform`
- path family: `linear`
- mode: `training_free`
- `k=0`
- solvers: `euler`, `heun2`, `stork4`
- compare: `uniform nodes` vs `solver-aware nodes`
- metric: `FID`
- NFE: `[6, 12, 18, 24, 30, 48, 96]`

## 6. 当前限制

- STORK4 仍然只完成了 non-uniform node support + benchmark pipeline；它的 monitor 仍是 Heun2 proxy heuristic。
- propagation-aware 扩展与 fixed-point continuation 的新实现，见 `experiments/solver_aware_clock_propagation.md`。
- solver-aware monitor / propagation profile 默认会在评估前做一次共享构造，并缓存为 `solver_aware_profile.pt` / `solver_aware_artifacts.pt`。
