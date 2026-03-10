# Task 003 - Paper Results Pipeline

## Metadata

- Task ID: 003
- Title: Paper Results Pipeline
- Slug: paper-results-pipeline
- Status: pending
- Type: refactor
- Priority: high
- Owner: agent
- Reviewer: pending
- Created At: 2026-03-10 23:18
- Updated At: 2026-03-10 23:18
- Branch: codex/refactor/t003-paper-results-pipeline
- Worktree: ../wt-t003-paper-results-pipeline
- Related Issues: N/A
- Related PRs: N/A
- Dependencies: 001-ft-clock-experiment-platform, 002-core-ft-clock-runtime
- Follow-up Tasks: none

---

## Goal

提供 YAML-first 实验编排、tidy results 记录、主图主表导出与实验 1-5 所需汇总能力。

---

## Background / Context

- `experiments/run_experiments.py` 当前只会拼旧 alpha sweep 命令。
- `experiments/visualize_results.py` 当前只接受 `alpha/epoch/nfe/fid` 结构，无法表达 path、clock、solver、metric。

---

## Scope

- 定义 YAML-first 配置 schema。
- 重写 runner 生成 train/eval/plot/report 任务。
- 重写 tidy CSV/JSONL 结果结构。
- 生成 FID-NFE 曲线、beta-NFE 热力图、主表、schedule family 表、cross-path 表。
- 为实验 1-5 提供示例配置。

---

## Non-goals

- 不改训练 runtime 核心数学。
- 不负责机制分析图。

---

## Approach

- 以结构化 YAML 描述 dataset/path/clock/solver/metrics/sweeps。
- 以 long-form results 文件驱动画图和表格导出。
- runner 负责命令展开、状态跟踪和结果汇总，不负责科学逻辑硬编码。

---

## Execution Plan

1. 定义配置 schema 与 canonical 字段。
2. 重写 runner 与结果落盘逻辑。
3. 重写 plot/report 脚本。
4. 增加实验 1-5 示例配置与 synthetic fixture 测试。

---

## Test Plan

- 配置展开测试。
- 结果 schema 测试。
- best-beta 选择与 cross-path 聚合测试。
- synthetic fixture 绘图回归测试。

---

## Risks

- 如果 schema 设计不稳，下游 analysis 会不断返工。
- 如果结果字段不统一，主表和热力图会难以复用。

---

## Open Questions

- None.

---

## Plan Self-Review

### Gaps

- 需要兼顾机器可读和论文导出两种消费方式。

### Feasibility

- 可行，主要集中在 `experiments/*`。

### Risks Review

- schema 一旦落地，应尽量避免再改字段名。

### Should This Task Be Split?

- No.

---

## Approval-Ready Summary

该子任务负责把 runtime 结果转成论文可用产物，是实验 1-5 的直接承载层。

---

## Progress Log

- 2026-03-10 23:18 — Task created with status: pending.

---

## Decisions

- 结果格式采用 tidy CSV/JSONL。

---

## Working Notes

### Files Touched

- tasks/003-paper-results-pipeline.md

### Notes

- none
