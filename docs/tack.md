# TACK

`tack` 是接在现有 `sampling_solver` 分发链路上的 training-free 采样器，只在 eval / sampling 阶段生效，不改训练目标。

## 运行方式

主评估：

```bash
python3 experiments/run_experiments.py --config experiments/configs/ft_clock/tack_solver_sensitivity.yaml
```

Ablation：

```bash
python3 experiments/run_experiments.py --config experiments/configs/ft_clock/tack_ablation.yaml
```

## 参数

- `--sampling_solver tack`
- `--tack_profile_grid_size`: 离线 profile 网格数，默认 `64`
- `--tack_profile_batch_size`: 每个 profile batch 的样本数，默认 `256`
- `--tack_profile_num_batches`: 每个网格点的 Monte Carlo batch 数，默认 `8`
- `--tack_profile_eps`: profile 数值 epsilon，默认 `1e-8`
- `--tack_lambda`: 误差等分布项系数，默认 `1.0`
- `--tack_eta`: stiff floor 系数，默认 `0.25`
- `--tack_profile_cache`: 是否启用 profile cache，默认 `true`
- `--tack_force_recompute_profile`: 是否强制重算 profile，默认 `false`
- `--tack_chi_lo`: AB3 / AB2 切换阈值，默认 `0.10`
- `--tack_chi_hi`: AB2 / Heun 切换阈值，默认 `0.50`
- `--tack_tau`: defect 目标值，默认 `0.05`
- `--tack_startup_steps`: 启动阶段 Heun 步数，默认 `2`
- `--tack_enable_dyadic`: 是否启用 dyadic 调步，默认 `true`
- `--tack_batch_shared_adapt`: 是否按 batch 共享自适应决策，默认 `true`
- `--tack_min_dr_scale`: 最小 dyadic 步长比例，默认 `0.25`
- `--tack_max_dr_scale`: 最大 dyadic 步长比例，默认 `4.0`
- `--tack_monitor_estimator`: `{auto, finite_diff, jvp}`，默认 `auto`
- `--tack_mode`: `{full, clock_only, online_only}`，默认 `full`

## NFE 口径

- `eval_nfe` / `requested_eval_nfe` 表示请求的真实网络前向预算
- `realized_nfe` 表示实际发生的网络前向次数
- 对 `tack`，实现采用 `1 + accepted_steps` 的缓存口径：
  - 初始 `g(z_0, 0)` 记 1 次前向
  - 每个 accepted step 只新增 1 次 `g(z_{n+1}^P, r_{n+1})`
  - 下一步历史斜率复用该 predictor-corrector 终点查询，不额外补一次 `g(z_{n+1}, r_{n+1})`

## Artifact

每个 `eval_ep*_nfe*` 目录下会输出：

- `tack_profile.pt`
- `tack_profile_cache.pt`
- `tack_profile.json`
- `tack_profile.csv`
- `tack_profile_debug.png`
- `tack_online_summary.json`
- `tack_online_steps.csv`
- `tack_online_debug.png`
- `solver_stats.json`

`tack_profile.json` 会包含：

- `s_grid`
- `q1_values`
- `q2_values`
- `q1_smoothed`
- `q2_smoothed`
- `rho_raw`
- `rho_star`
- `phi_values`
- `psi_query_grid`
- `psi_values`
- `psi_prime_values`

## 机制分析

`experiments/analyze_mechanisms.py` 现在支持 `sampling_solver: tack`。当配置里选择 `tack` 时，会在分析输出目录下额外写出每个 NFE 对应的 `tack_profiles/` artifact。

## 当前限制

- 当前只实现了 `tack_batch_shared_adapt=true` 的 batch-shared 自适应版本
- `tack_monitor_estimator=auto` 当前解析为 `finite_diff`
- `solver_stats.json` 会聚合整个 eval 过程的标量统计，但 `tack_online_steps.csv` 仍然对应单次采样轨迹
