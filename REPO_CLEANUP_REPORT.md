# 仓库梳理与裁剪说明

## 1) 目录功能梳理

### 保留目录

- `examples/image/`
  - 你当前科研项目的核心代码：数据加载、训练循环、评估采样、模型定义、命令行参数。
- `flow_matching/`
  - `examples/image` 依赖的核心算法组件。
  - `path/`：概率路径与调度相关实现。
  - `solver/`：ODE / 离散求解器。
  - `utils/`：模型封装等工具。
  - `loss/`：库中的损失定义（目前保留以避免潜在依赖断裂）。

### 已移除目录/文件（与当前项目主线无关）

- `.github/`：CI、issue 模板、自动化流程。
- `docs/`：文档站点源码。
- `tests/`：单元测试与集成测试。
- `assets/`：README 与文档展示素材。
- `examples/text/`：文本任务示例，不属于当前图像研究主线。
- `examples/` 下 image 以外的教学 notebook 与说明文件。

## 2) 裁剪原则

- **只保留论文复现实验需要的最小闭环**：`examples/image + flow_matching + 基础打包文件`。
- **优先保证可运行性**：保留 `flow_matching` 全库，避免过度删除导致隐式 import 断裂。
- **去除非研究主线资产**：文档、CI、测试、跨任务示例全部剔除。

## 3) 后续建议

- 当论文实验完全稳定后，可继续做第二阶段瘦身：
  1. 基于 import 图继续裁掉 `flow_matching` 中未用模块；
  2. 增加最小 smoke test（1 batch train + 1 batch eval）用于开源自检；
  3. 在 `examples/image/README.md` 中补齐论文配置与复现命令。
