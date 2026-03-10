# Task 004 - Mechanism Analysis Suite

## Metadata

- Task ID: 004
- Title: Mechanism Analysis Suite
- Slug: mechanism-analysis-suite
- Status: pending
- Type: refactor
- Priority: high
- Owner: agent
- Reviewer: pending
- Created At: 2026-03-10 23:18
- Updated At: 2026-03-10 23:18
- Branch: codex/refactor/t004-mechanism-analysis-suite
- Worktree: ../wt-t004-mechanism-analysis-suite
- Related Issues: N/A
- Related PRs: N/A
- Dependencies: 001-ft-clock-experiment-platform, 002-core-ft-clock-runtime, 003-paper-results-pipeline
- Follow-up Tasks: none

---

## Goal

提供实验 6-9 所需的离线分析能力，包括 beta-NFE 热力图、速度/损失分布、轨迹预算分配与 solver sensitivity。

---

## Background / Context

- 当前仓库没有机制分析脚本。
- 论文主张依赖 terminal focusing–stiffness tradeoff，仅报 FID 不足以支撑。

---

## Scope

- 构建离线 checkpoint 分析脚本。
- 计算 `||\\tilde b_r||`、loss density、末端 20% 区间状态变化量、平均步长与末端步长比例。
- 生成 linear 必选热力图与 solver sensitivity 报告。
- 为实验 6-9 提供配置模板与产物导出。

---

## Non-goals

- 不负责主训练 runtime 重构。
- 不负责主实验 1-5 的 runner 与主图主表。

---

## Approach

- 分析统一走离线脚本，只读取 checkpoint、配置与少量样本。
- 重用 runtime 的 path/clock/solver 逻辑，避免分析口径与训练口径不一致。

---

## Execution Plan

1. 定义分析输入与输出 schema。
2. 实现 trajectory/stat extraction。
3. 实现 heatmap、loss density、budget allocation、solver sensitivity 图表。
4. 补分析 fixture 与 smoke test。

---

## Test Plan

- synthetic trajectory 统计测试。
- 分析输出 schema 测试。
- 机制图表生成 smoke test。

---

## Risks

- 轨迹与中间量如果存储过密会导致磁盘压力过大。
- 分析口径如果没有复用 runtime，会出现训练/分析不一致。

---

## Open Questions

- None.

---

## Plan Self-Review

### Gaps

- 需要在统计稳定性与存储成本之间做平衡，但已决定采用离线抽样。

### Feasibility

- 可行，前提是 002 提供统一 runtime API。

### Risks Review

- 机制分析必须严格基于与主结果相同的 NFE 和 solver 定义。

### Should This Task Be Split?

- No.

---

## Approval-Ready Summary

该子任务负责把“有效”进一步解释成“为什么有效”，直接支撑论文机制部分。

---

## Progress Log

- 2026-03-10 23:18 — Task created with status: pending.

---

## Decisions

- 机制分析固定为离线脚本，不在训练期长期记录。

---

## Working Notes

### Files Touched

- tasks/004-mechanism-analysis-suite.md

### Notes

- none
