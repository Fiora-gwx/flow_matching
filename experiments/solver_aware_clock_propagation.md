# Propagation-Aware Solver-Aware Clock

## 1. 原理

在现有 solver-aware 支路上，我们继续保留局部截断误差 monitor：

- Euler: `Q_E(s) = E_{z~p_s} ||L_u u(z,s)||^2`
- Heun2: `Q_H(s) = E_{z~p_s} ||L_u^2 u(z,s)||^2`

并额外加入传播上界：

`G(s) = exp(int_s^1 ell(t) dt)`

其中 `ell(s)` 是沿路径样本的 Jacobian spectral envelope，用来上界后续误差传播放大。

## 2. Euler 误差递推

Euler 的离散误差满足：

`||e_{n+1}|| <= (1 + h_n ell_n)||e_n|| + C_E M_E(s_n) h_n^2`

展开后得到：

`||e_N|| <= C_E sum_i G_i M_E(s_i) h_i^2`

连续代理写成：

`J_E[rho] = int G(s) M_E(s) rho(s)^(-1) ds`

最优密度为：

`rho_E*(s) propto [G(s) M_E(s)]^(1/2)`

当 `M_E(s) = sqrt(Q_E(s))` 时：

`rho_E*(s) propto G(s)^(1/2) Q_E(s)^(1/4)`

## 3. Heun2 扩展

Heun2 仍以 `Q_H(s) = E||L_u^2 u||^2` 作为局部 monitor，传播感知密度使用：

`rho_H*(s) propto G(s)^(1/3) Q_H(s)^(1/6)`

当前实现中：

- propagation-aware 分支统一记为 empirical / heuristic proxy，而不是 strict theorem-backed
- `spectral_q95` 更平滑，当前推荐作为训练-free 对比配置
- `stork4` 仍然只把 propagation-aware nodes 当作 phase-2 heuristic extension，因此 `solver_aware_theorem_backed=false`

## 4. G(s) 的实现

默认传播实现为 `solver_aware_g_mode=jacobian_envelope`：

1. 在固定参考批次上沿 `s-grid` 扫描
2. 对每个 `z~p_s` 用 power iteration + JVP/VJP 估计 `||J_x u(z,s)||_2`
3. 按 `solver_aware_g_estimator` 聚合得到 `raw_ell(s_j)`
4. 用 `max-pool1d(radius=solver_aware_g_pool_radius)` 构造 `env_ell(s_j)`
5. `hat_ell(s_j) = solver_aware_g_safety_factor * env_ell(s_j)`
6. `hat_G(s_j) = exp(sum_{m>=j} hat_ell(s_m) Delta s_m)`

这里的尾积分采用右端点 Riemann sum，但仍然只把它当作 empirical propagation proxy：

- `ell(s)` 来自有限 batch 的 Jacobian spectral 估计
- power iteration 只做有限轮近似
- pooling / smoothing / 数值积分都会引入经验近似

因此代码中不会再把 propagation-aware 分支标成 strict theorem-backed。

为了保持曲线连续性并避免之前的高方差问题：

- monitor 与 propagation 都使用固定 reference batch 沿 `s` 扫描
- JVP/VJP 路径显式关闭 autocast
- 所有统计都按 micro-batch detach 后再累计，避免显存图堆积

## 5. Fixed-Point 微调

`solver_aware_clock_mode=fixed_point` 且 `solver_aware_k>=1` 时，不再从头训练，而是做 continuation finetuning：

1. 第 0 轮加载 base checkpoint
2. 每轮先用当前 checkpoint 估计 `Q/G`
3. 构造新的 continuous solver-aware profile
4. 训练阶段按这条 profile 采样时间，而不是只在 eval 阶段换节点
5. 继续微调 `solver_aware_finetune_epochs`
6. 保存 `iter_k` checkpoint

训练时间采样仍然复用现有 `time_sampling_strategy` 语义：

- `uniform` 仍在 solver 时间 `r` 上均匀采样
- `ds_dr_sq` / `mixed_lambda` / `stratified_mixed` 则使用新的 arbitrary clock importance cdf

## 6. 新参数

- `--solver_aware_use_propagation`
- `--solver_aware_g_mode {none,jacobian_envelope}`
- `--solver_aware_g_estimator {spectral_max,spectral_maxpool,spectral_q95}`
- `--solver_aware_g_power_iters`
- `--solver_aware_g_pool_radius`
- `--solver_aware_g_safety_factor`
- `--solver_aware_g_cache_path`
- `--solver_aware_finetune_epochs`
- `--solver_aware_finetune_lr`
- `--solver_aware_finetune_reset_optimizer`
- `--solver_aware_finetune_resume_from_previous`

## 7. 推荐运行

Training-free propagation-aware eval:

`python3 experiments/run_experiments.py --config experiments/configs/ft_clock/solver_aware_propagation_training_free_linear_uniform.yaml`

可视化时复用 solver-sensitivity baseline：

`python3 experiments/visualize_solver_aware_results.py --results_dir experiments/results/ft_clock_solver_aware_propagation_training_free_linear_uniform --artifact_group ft_clock_solver_aware_propagation_training_free_linear_uniform --baseline_csv experiments/results/ft_clock_solver_sensitivity_uparam/results.csv`
