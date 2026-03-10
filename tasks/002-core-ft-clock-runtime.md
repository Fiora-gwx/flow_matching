# Task 002 - Core FT-Clock Runtime

## Metadata

- Task ID: 002
- Title: Core FT-Clock Runtime
- Slug: core-ft-clock-runtime
- Status: pending
- Type: refactor
- Priority: high
- Owner: agent
- Reviewer: pending
- Created At: 2026-03-10 23:18
- Updated At: 2026-03-10 23:18
- Branch: codex/refactor/t002-core-ft-clock-runtime
- Worktree: ../wt-t002-core-ft-clock-runtime
- Related Issues: N/A
- Related PRs: N/A
- Dependencies: 001-ft-clock-experiment-platform
- Follow-up Tasks: none

---

## Goal

提供通用的 continuous-image `path + clock + solver budget + dataset` 运行时，使训练、评估、分析都不再依赖旧 FT-EqM/NT-FT-FM 特例逻辑。

---

## Background / Context

- `examples/image/train_arg_parser.py` 当前在 import 时就依赖 `torchdiffeq`，环境未装时主入口直接失效。
- `train_loop.py` 和 `eval_loop.py` 当前将 linear FM、FT-EqM、NT-FT-FM 写成并列分支，无法表达通用 path/clock。
- `train.py` 当前不支持 `cifar100`。

---

## Scope

- 去掉 continuous 主线对 `torchdiffeq` 的入口级硬依赖。
- 实现 `linear` 与 `trig_vp` path。
- 实现 `uniform`、`ft_linear_beta`、`ft_vp_beta`、`poly_a0.5`、`poly_a2.0`、`cosine`、`sigmoid_k8`、`exp_l3` clock。
- 实现 `euler` 与 `heun2` 的固定步长采样和真实 NFE 计数。
- 为 `cifar100` 增加数据集入口。
- 为 path/clock/NFE 增加最小测试。

---

## Non-goals

- 不负责结果聚合与画图。
- 不负责实验矩阵 sweep 编排。
- 不负责机制分析脚本。

---

## Approach

- 新建独立的 training runtime 模块，封装 path、clock、target velocity、sampling trajectory、NFE accounting。
- 在 `train_loop.py` 和 `eval_loop.py` 中只保留统一 runtime 调用，不保留旧 FT-EqM/NT-FT-FM 分支。
- 用固定步长 Euler/Heun2 替代主线对 `torchdiffeq` 的依赖，确保环境未装时主实验仍可运行。

---

## Execution Plan

1. 抽离并实现 path/clock 数学模块。
2. 改造 parser 与 train/eval 入口，接入新 runtime。
3. 加入 `cifar100` 数据集支持。
4. 为 fixed-step solver 加入真实 NFE 计数与轨迹返回能力。
5. 补最小单元测试与 smoke checks。

---

## Test Plan

- 时钟端点与单调性测试。
- `linear`/`trig_vp` 目标速度有限差分测试。
- Euler/Heun2 实际 NFE 计数测试。
- `cifar100` 配置入口 smoke test。

---

## Risks

- 旧连续主线删除后，历史 alpha-sweep 命令会失效。
- 固定步长 Heun2 的实现如果与 NFE 预算换算不一致，会影响全链路结果。

---

## Open Questions

- None.

---

## Plan Self-Review

### Gaps

- 需要同时兼顾训练、评估和分析三方复用，runtime 接口命名必须尽量稳定。

### Feasibility

- 可行，代码集中在 `examples/image/training/*` 和 parser 层。

### Risks Review

- 需要严格测试数学公式与 NFE 计数，否则下游分析都会被污染。

### Should This Task Be Split?

- No.
- 该子任务本身已经是最小可执行切片。

---

## Approval-Ready Summary

先打通统一 runtime，是后续所有 YAML runner、图表和分析脚本的前提。

---

## Progress Log

- 2026-03-10 23:18 — Task created with status: pending.

---

## Decisions

- 该子任务优先解除 `torchdiffeq` 入口阻塞。

---

## Working Notes

### Files Touched

- tasks/002-core-ft-clock-runtime.md

### Notes

- none
