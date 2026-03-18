# FT-Clock Image Experiment Platform

这个仓库是基于 `flow_matching` 改造的论文实验代码，当前主线已经迁移到通用的 `path + clock + solver` 语义，用于复现 FT-Clock 在 CIFAR-10 / CIFAR-100 上的主结果、跨路径对比和机制分析。

## 当前主入口

- 训练入口：`examples/image/train.py`
- YAML 实验入口：`experiments/run_experiments.py`
- 最优 beta 物化：`experiments/resolve_best_betas.py`
- 主图主表：`experiments/visualize_results.py`
- 机制分析：`experiments/analyze_mechanisms.py`

## 参数口径

连续实验不再使用旧的 `alpha/use_ft_eqm/use_nt_ft_fm`。

当前有效参数是：

- `--path_family {linear,trig_vp}`
- `--clock_family {uniform,ft_linear_beta,ft_vp_beta,poly_a0.5,poly_a2.0,cosine,sigmoid_k8,exp_l3}`
- `--clock_beta <beta>`
- `--sampling_solver {euler,heun2}`
- `--eval_nfe <real network forward calls>`

`experiments/run_experiments.py` 会显式拒绝 legacy config key，避免旧配置静默误用。

## 环境安装

```bash
conda env create -f environment.yml
conda activate flow_matching
pip install -e .
cd examples/image
pip install -r requirements.txt
cd ../..
```

当前实验和测试依赖至少包括：`pytorch`、`torchvision`、`torchmetrics[image]`、`matplotlib`、`pyyaml`、`scikit-learn`。

当前支持的评估指标包括：`fid`、`precision_recall`、`inception_score`。

## 标准实验命令

### E1 Linear Main

```bash
python3 experiments/run_experiments.py --config experiments/configs/ft_clock/linear_main.yaml
python3 experiments/visualize_results.py \
  --csv experiments/results/ft_clock_linear_main/results.csv \
  --out experiments/results/ft_clock_linear_main/plots \
  --artifact_group ft_clock_linear_main
```

如果历史结果里只有 `fid` / `precision_recall`，现在想补 `inception_score`，直接重跑同一个 YAML 即可。`experiments/run_experiments.py` 会跳过已完成训练，只对缺失指标执行 `eval_only` 回填。

YAML runner 的训练阶段默认会传 `--eval_frequency -1`，避免每 50 个 epoch 做一次中途 FID/IS；最终 checkpoint 仍会保存，真正的多 NFE 评估由 runner 在训练完成后统一执行。

只生成 E6 热力图：

```bash
python3 experiments/visualize_results.py \
  --csv experiments/results/ft_clock_linear_main/results.csv \
  --out experiments/results/ft_clock_linear_main/plots_heatmap \
  --artifact_group ft_clock_linear_main \
  --plot_heatmap_only
```

### E2 / E3 / E5 自动回填 best beta

以下配置不再手工写死 beta，而是从 E1 或 E4 的结果中自动解析：

- `experiments/configs/ft_clock/cifar100_transfer.yaml`
- `experiments/configs/ft_clock/schedule_family.yaml`
- `experiments/configs/ft_clock/cross_path.yaml`

直接运行：

```bash
python3 experiments/run_experiments.py --config experiments/configs/ft_clock/cifar100_transfer.yaml
python3 experiments/run_experiments.py --config experiments/configs/ft_clock/schedule_family.yaml
python3 experiments/run_experiments.py --config experiments/configs/ft_clock/cross_path.yaml
```

如需把解析后的 beta 固化成可归档 YAML：

```bash
python3 experiments/resolve_best_betas.py \
  --config experiments/configs/ft_clock/cross_path.yaml \
  --out experiments/configs/ft_clock/generated/cross_path_resolved.yaml
```

### E4 VP Path

```bash
python3 experiments/run_experiments.py --config experiments/configs/ft_clock/trig_vp.yaml
```

### E7-E9 Mechanism / Solver Analysis

```bash
python3 experiments/run_experiments.py --config experiments/configs/ft_clock/solver_sensitivity.yaml
python3 experiments/analyze_mechanisms.py --config experiments/configs/ft_clock/mechanism_analysis.yaml
```

`mechanism_analysis.yaml` 支持：

- `checkpoint_epoch`: 解析指定 epoch checkpoint
- `checkpoint_path`: 直接指定 checkpoint 文件

## 配置说明

- `experiments/configs/ft_clock/linear_main.yaml`: E1 主 sweep，同时产出 E6 热力图数据
- `experiments/configs/ft_clock/cifar100_transfer.yaml`: E2，使用 E1 自动解析出的 linear FT-best beta
- `experiments/configs/ft_clock/schedule_family.yaml`: E3，使用 E1 自动解析出的 linear FT-best beta
- `experiments/configs/ft_clock/trig_vp.yaml`: E4
- `experiments/configs/ft_clock/cross_path.yaml`: E5，linear FT-best 来自 E1，VP FT-best 来自 E4
- `experiments/configs/ft_clock/mechanism_analysis.yaml`: E7/E8
- `experiments/configs/ft_clock/solver_sensitivity.yaml`: E9

## 说明

- 结果聚合默认先按 seed 求 `mean/std`，再选择 best beta；不再按单次最小 FID 取最优。
- `eval_nfe` 口径是“真实网络前向调用次数”。CFG 打开时会按两次真实前向计数。
- 评估会循环采样直到达到目标 `fid_samples`，并把 `real_samples/synthetic_samples` 写回结果行。
