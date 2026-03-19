# Image Experiments

`examples/image` 是当前论文实验的训练和评估入口。

## 安装

```bash
conda activate flow_matching
cd examples/image
pip install -r requirements.txt
cd ../..
```

## 单实验训练

连续 FT-Clock 训练示例：

```bash
python3 examples/image/train.py \
  --dataset cifar10 \
  --data_path ./data/cifar10 \
  --path_family linear \
  --clock_family ft_linear_beta \
  --clock_beta 0.5 \
  --sampling_solver heun2 \
  --cfg_scale 0.0 \
  --class_drop_prob 1.0 \
  --epochs 500 \
  --batch_size 128 \
  --eval_frequency 50 \
  --compute_fid
```

VP path 训练示例：

```bash
python3 examples/image/train.py \
  --dataset cifar10 \
  --data_path ./data/cifar10 \
  --path_family trig_vp \
  --clock_family ft_vp_beta \
  --clock_beta 0.5 \
  --sampling_solver heun2 \
  --cfg_scale 0.0 \
  --class_drop_prob 1.0
```

## YAML-first 实验

推荐优先用 `experiments/configs/ft_clock/*.yaml`，而不是手写长命令：

```bash
python3 experiments/run_experiments.py --config experiments/configs/ft_clock/linear_main.yaml
python3 experiments/run_experiments.py --config experiments/configs/ft_clock/trig_vp.yaml
python3 experiments/run_experiments.py --config experiments/configs/ft_clock/solver_sensitivity.yaml
```

如果某个实验组已经训练完成，后续只是在 YAML 的 `metrics` 中新增了 `inception_score`，重新执行同一个 `run_experiments.py --config ...` 即可；runner 会复用现有 checkpoint，只回填缺失评估，不会重新训练。

其中：

- E2 / E3 / E5 会自动从 E1 / E4 结果解析 best beta
- E3 / E5 / E9 会在配置允许时直接复用 E1 / E4 已训练 checkpoint，避免重复训练
- 可用 `experiments/resolve_best_betas.py` 输出一份显式 resolved YAML

## 评估与可视化

```bash
python3 experiments/visualize_results.py \
  --csv experiments/results/ft_clock_linear_main/results.csv \
  --out experiments/results/ft_clock_linear_main/plots \
  --artifact_group ft_clock_linear_main
```

只导出 beta-NFE 热力图：

```bash
python3 experiments/visualize_results.py \
  --csv experiments/results/ft_clock_linear_main/results.csv \
  --out experiments/results/ft_clock_linear_main/plots_heatmap \
  --artifact_group ft_clock_linear_main \
  --plot_heatmap_only
```

## 机制分析

```bash
python3 experiments/analyze_mechanisms.py --config experiments/configs/ft_clock/mechanism_analysis.yaml
```

`mechanism_analysis.yaml` 支持两种 checkpoint 选择方式：

- `checkpoint_from`
- `checkpoint_epoch`
- `checkpoint_path`

## 采样过程图与轨迹图

默认直接读取前序实验结果：

```bash
python3 experiments/plot_sampling_progression.py
python3 experiments/plot_particle_trajectory_comparison.py
```

## 结果口径

- `eval_nfe` 表示真实网络前向次数
- `results.csv` 会记录 `real_samples` 和 `synthetic_samples`
- 主表和热力图先按 seed 聚合 `mean/std`，再选 best beta
