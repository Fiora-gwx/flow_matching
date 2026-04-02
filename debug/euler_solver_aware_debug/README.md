# Euler Solver-Aware Debug

这个目录只服务于一个目的：定位“为什么 training-free solver-aware clock 对 Euler 全面变差”。

设计原则：

- 只做 `Euler`，不扩展到 `Heun2 / STORK4`
- 默认不改主项目训练/评估入口
- 只读复用现有 checkpoint、monitor、clock、node、solver、FID 原语
- 所有输入、缓存、图、CSV、summary 都放在这个目录下
- 调试完成后直接删除 `debug/euler_solver_aware_debug/` 即可回滚

## 理论背景

边缘 ODE:

```text
dx/ds = u(x,s)
```

Euler 的局部截断误差主项由下面这个 material derivative 控制：

```text
L_u u = ∂_s u + J_x u · u
```

平方型 Euler monitor:

```text
Q_E(s) = E_{z ~ p_s} ||L_u u(z,s)||^2
```

当前 training-free solver-aware Euler 使用的连续密度为：

```text
rho_E(s) = dr/ds ∝ (Q_E(s) + eps)^(1/4)
```

累计时钟：

```text
phi_E(s) = ∫_0^s (Q_E(u)+eps)^(1/4) du / ∫_0^1 (Q_E(u)+eps)^(1/4) du
```

离散节点由下面的反演给出：

```text
s_n = psi_E(n / N),   psi_E = phi_E^{-1}
```

这次 debug 的重点不是“让 Euler 一定变好”，而是检查下面这些点是否解释了 FID 恶化：

- `Q_E(s)` 是否本身就不合理或数值不稳定
- smoothing 前后差异是否很大
- `phi_E(s)` 是否过于陡峭
- nodes 是否过度聚焦在两端或终端
- 对 Euler 来说是否造成中段大步长
- coarse monitor grid 离散后是否显著偏离连续目标
- JVP monitor 是否天然噪声更大
- 问题更像 monitor / density / node allocation 错配，还是更像实现 bug

## 目录结构

```text
debug/euler_solver_aware_debug/
├── README.md
├── configs/
│   └── default.yaml
├── outputs/
├── monitor_debug.py
├── plot_debug.py
├── compare_variants.py
└── run_euler_debug.py
```

文件职责：

- `run_euler_debug.py`: 串联完整 debug 流程
- `monitor_debug.py`: 计算/缓存 `Q_E`、smoothing、clipping、density、`phi`、nodes、稳定性分析
- `plot_debug.py`: 输出 monitor / density / phi / nodes / step diagnostics 图
- `compare_variants.py`: 只对 Euler 跑 uniform 与多种 debug 变体的 FID 对比

## 运行方式

默认运行：

```bash
python3 debug/euler_solver_aware_debug/run_euler_debug.py \
  --dataset cifar10 \
  --nfe_list 6 12 18 24 30 48 96 \
  --checkpoint_from_artifact_group ft_clock_solver_aware_training_free_linear_uniform \
  --checkpoint_from_source_exp_name linear_uniform_euler_baseline \
  --checkpoint_from_epoch 499
```

如果自动解析 checkpoint 不通，可以直接传路径：

```bash
python3 debug/euler_solver_aware_debug/run_euler_debug.py \
  --dataset cifar10 \
  --nfe_list 6 12 18 24 30 48 96 \
  --checkpoint_path /abs/path/to/checkpoint-499.pth
```

也可以从默认配置启动：

```bash
python3 debug/euler_solver_aware_debug/run_euler_debug.py \
  --config debug/euler_solver_aware_debug/configs/default.yaml
```

## 运行依赖

这个 debug 工具链会复用主项目的真实运行依赖，至少需要：

- `torch`
- `torchvision`
- `torchmetrics`
- `pyyaml`
- `matplotlib`

当前 worktree 如果缺依赖，脚本会直接报清楚缺什么，不会静默降级。

## 输出内容

默认输出都写到：

```text
debug/euler_solver_aware_debug/outputs/
```

关键产物包括：

- `run_config_resolved.json`: 本次 debug 的最终配置快照
- `checkpoint_resolution.json`: checkpoint 解析结果，包含实际 path 和来源字段
- `euler_debug_results.csv`: 各 NFE / 各变体 / 各诊断字段 / FID
- `euler_debug_summary.md`: 自动总结
- `profiles/...`: raw / smoothed / clipped / density / phi / nodes / step diagnostics 的 JSON / CSV / 图
- `stability/...`: batch size / seed 稳定性统计与 mean ± std 图
- `grid_sweep/...`: `monitor_grid_size` 对 `phi / nodes / FID` 的影响

## 当前默认会覆盖的检查项

- 基础数值自检
- monitor / clock / node 可视化
- smoothing 对比
- clipping 对比
- mixed density 对比
- min/max step 约束对比
- continuous-to-discrete grid-size sweep
- monitor 稳定性检查
- 关键区间自动定位
- 只针对 Euler 的 FID 对比

## 删除方式

这个目录是隔离调试工具链，不会修改主流程默认行为。

删除方式：

```bash
rm -rf debug/euler_solver_aware_debug
```
