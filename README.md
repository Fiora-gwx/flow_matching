# NT-FT-FM Research Codebase (Based on flow_matching)

这个仓库是基于教学库 `flow_matching` 裁剪后的科研代码版本，目标是专注于图像任务上的 Flow Matching / FT-EqM / NT-FT-FM 实验。

## 保留内容

- `examples/image/`：训练、评估、采样、模型定义与参数解析（你的主要实验入口）。
- `flow_matching/`：`examples/image` 运行依赖的核心库（path / solver / utils 等）。
- `setup.py`、`environment.yml`：基础安装与环境说明。
- `LICENSE`：许可证。

## 已裁剪内容

为了便于论文开源发布，已移除与当前科研目标无关的模块：

- 文档与站点：`docs/`
- CI 与 issue 模板：`.github/`
- 测试集：`tests/`
- 文本任务示例：`examples/text/`
- 2D 与 standalone 教学 notebook 示例（非 image 主线）
- 仓库宣传素材：`assets/`

## 快速开始

```bash
conda env create -f environment.yml
conda activate flow_matching
pip install -e .
python examples/image/train.py --help
```

## 说明

- NT-FT-FM 入口参数：`--use_nt_ft_fm`、`--kappa`（与 `--use_ft_eqm` 互斥）。
- 当前仓库聚焦论文复现实验主路径，若后续需要恢复教学内容，可从上游仓库对应版本重新同步。
