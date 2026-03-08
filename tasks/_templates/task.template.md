# Task NNN - <Task Title>

<!--
说明：
- 文件命名：tasks/NNN-slug.md
- 例如：tasks/015-billing-page-filter.md
- 字段名保持英文，说明与内容可中英混排
- 本文件是单个任务的唯一主记录文件
-->

## Metadata

- Task ID: NNN
- Title: <short task title>
- Slug: <kebab-case-slug>
- Status: pending
- Type: <feat | fix | chore | refactor>
- Priority: <low | medium | high>
- Owner: <agent / person>
- Reviewer: <agent / person / pending>
- Created At: <YYYY-MM-DD HH:mm>
- Updated At: <YYYY-MM-DD HH:mm>
- Branch: <feat|fix|chore|refactor>/tNNN-slug
- Worktree: ../wt-tNNN-slug
- Related Issues: <issue links or N/A>
- Related PRs: <PR links or N/A>
- Dependencies: <none / task ids / external dependency>
- Follow-up Tasks: <none / task ids>

---

## Goal

<!--
说明：
- 用 1~3 句话写清“要解决什么问题”
- 不写实现细节，只写目标结果
-->

<Describe the task goal clearly.>

---

## Background / Context

<!--
说明：
- 为什么要做这个任务
- 当前现状、问题来源、业务背景、代码背景
- 可写引用的模块、页面、接口、历史问题
-->

<Describe background and context.>

---

## Scope

<!--
说明：
- 明确本任务“包含什么”
- 尽量写成可验证、可交付的事项
-->

- <in-scope item 1>
- <in-scope item 2>
- <in-scope item 3>

---

## Non-goals

<!--
说明：
- 明确本任务“不做什么”
- 防止执行阶段任务膨胀
-->

- <out-of-scope item 1>
- <out-of-scope item 2>

---

## Approach

<!--
说明：
- 说明打算如何做
- 可以写技术路径、架构思路、关键实现策略
- 不要求写到代码级别，但要足够让 reviewer 判断方案是否可行
-->

<Describe the proposed approach.>

---

## Execution Plan

<!--
说明：
- 拆成可执行步骤
- 每一步尽量独立、可检查、可回滚
-->

1. <step 1>
2. <step 2>
3. <step 3>
4. <step 4>

---

## Test Plan

<!--
说明：
- 写清如何验证结果
- 包括单测、集成测试、手工验证、构建验证、边界情况验证
-->

- Unit tests:
  - <test item>
- Integration / E2E:
  - <test item>
- Manual checks:
  - <test item>
- Build / lint / type checks:
  - <command or check item>

---

## Risks

<!--
说明：
- 写潜在风险、兼容性问题、依赖风险、回归风险
- 风险不怕多，关键是提前暴露
-->

- <risk 1>
- <risk 2>

---

## Open Questions

<!--
说明：
- 当前还不确定、需要决策、需要确认的信息
- 如果没有，写 None
-->

- <question 1>
- <question 2>

---

## Plan Self-Review

<!--
说明：
- 这是 planning 阶段必须完成的自检
- 判断方案是否有缺口、是否可落地、是否需要拆任务
-->

### Gaps

- <possible gap 1>
- <possible gap 2>

### Feasibility

- <feasibility assessment>

### Risks Review

- <risk review note>

### Should This Task Be Split?

- <yes/no + reason>

---

## Approval-Ready Summary

<!--
说明：
- planning 阶段最后给决策者看的简明摘要
- 只总结目标、范围、方案、风险、测试
- 不能开始实现
-->

<Concise summary for approval before execution.>

---

## Progress Log

<!--
说明：
- 执行阶段持续更新
- 每条记录必须带时间戳
- 每完成一个子任务就记录：完成了什么、改了哪些文件、达成了什么、发现了什么
-->

- <YYYY-MM-DD HH:mm> — Task created with status: pending.
- <YYYY-MM-DD HH:mm> — <progress note>

---

## Decisions

<!--
说明：
- 记录关键实现决策、权衡、范围变化
- 让 review / future reader 知道为什么这样做
-->

- <decision 1>
- <decision 2>

---

## Working Notes

<!--
说明：
- 记录执行期的重要上下文
- 可写 touched files、调试现象、接口行为、限制条件
-->

### Files Touched

- <file 1>
- <file 2>

### Notes

- <note 1>
- <note 2>

---

## Scope Updates

<!--
说明：
- 如果执行中发现任务膨胀、边界变化、拆分后续任务，在这里记录
- 没有可写 None
-->

- None

---

## Blockers

<!--
说明：
- 如果 blocked，必须记录：
- blocker 是什么
- 已尝试什么
- 解锁条件是什么
-->

- None

---

## Review Summary

<!--
说明：
- review 阶段填写
- 只评审，不修复
- 概述已完成内容、已验证内容、仍不确定内容
-->

### Completed

- <completed item 1>
- <completed item 2>

### Validated

- <validated item 1>
- <validated item 2>

### Uncertain

- <uncertain item 1>
- <uncertain item 2>

---

## Review Issues

<!--
说明：
- review 阶段必须使用这个表格格式
- 不要在本阶段直接修复
-->

| # | 问题描述 | 风险等级 | 影响范围（涉及哪些文件/模块） | 推荐修复方式 |
|---|---|---|---|---|
| 1 | <issue description> | <low/medium/high> | <files/modules> | <recommended fix> |

---

## Fix Log

<!--
说明：
- fix 阶段填写
- 仅记录用户明确批准修复的问题
-->

- <YYYY-MM-DD HH:mm> — Fixed review issue #<n>. Files: <files>. Impact: <impact>.

---

## Completion Summary

<!--
说明：
- done 阶段填写最终完成结果
-->

<Describe what was completed in the final state.>

---

## Completion Stats

<!--
说明：
- done 阶段填写统计信息
-->

- Branch: <branch name>
- Commits: <count or N/A>
- Files Changed: <count>
- Tests Run: <list>
- Review Issues Fixed: <count>
- Remaining Known Limitations: <summary or None>

---

## Final Commit

<!--
说明：
- 如果有最终提交信息，记录 commit hash + message
-->

- <commit hash> <commit message>

---

## Final Handoff

<!--
说明：
- done 阶段给最终交接使用
-->

- Archive Path: tasks/_archive/NNN-slug.md
- Suggested Final Commit / Merge Message: <message>
- Status: done
