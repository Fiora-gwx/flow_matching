# Repository Task Lifecycle

<!--
说明：
- 本文件定义任务在仓库中的标准生命周期
- 所有 agent 都必须遵守
- 任务阶段统一为：
  1. Plan
  2. Execute
  3. Review
  4. Fix
  5. Done
-->

---

## Core Principles

1. 任何代码改动前，必须先完成任务计划。
2. task 文件先出现在 main，便于并行任务与其他 agent 可见。
3. 真正的实现工作只允许在对应 worktree / branch 中进行。
4. task 文件是唯一事实来源（single source of truth）。
5. 所有关键节点必须写入 task 文件并带时间戳。
6. 不允许静默扩 scope。
7. review 阶段只审查，不修复。
8. fix 阶段只修用户明确批准的问题。
9. done 阶段只做收尾闭环，不做顺手新功能。
10. 每个安全阶段都应提醒可以提交 checkpoint。

---

## File and Naming Conventions

### Task File

Pattern:

`tasks/NNN-slug.md`

Examples:

- `tasks/015-billing-page-filter.md`
- `tasks/016-refactor-api-client.md`

Rules:

- `NNN` uses three digits
- increment sequentially
- `slug` uses kebab-case
- task name should reflect the actual problem or deliverable

### Branch Naming

Derived from task id and slug:

- `feat/tNNN-slug`
- `fix/tNNN-slug`
- `chore/tNNN-slug`
- `refactor/tNNN-slug`

Examples:

- `feat/t015-billing-page-filter`
- `fix/t016-api-timeout`
- `chore/t017-ci-cleanup`
- `refactor/t018-auth-context`

### Worktree Naming

Recommended location outside the main repository:

- `../wt-t015-billing-page-filter`
- `../wt-t016-api-timeout`

---

## Stage 1 — Plan

### Goal

Define task boundaries, implementation direction, validation plan, and risk profile before any code change.

### Required Steps

1. Read the relevant codebase area first.
2. Confirm current branch is `main`.
3. If not on `main`, switch and update:
   - `git checkout main`
   - `git pull`
4. Prepare a task plan in `tasks/NNN-slug.md` with at least:
   - Goal
   - Background / Context
   - Scope
   - Non-goals
   - Approach
   - Execution Plan
   - Test Plan
   - Risks
   - Open Questions
5. Perform a self-review of the plan:
   - gaps
   - feasibility
   - risks
   - whether the task should be split
6. Write the task with `status: pending`.
7. Do not implement code yet.
8. End with an approval-ready summary.
9. Wait for confirmation before execution.

### Plan Stage Output

- Task file created on `main`
- Task status is `pending`
- No implementation changes yet

### Recommended Commit
```bash
git add tasks/NNN-slug.md
git commit -m "chore(task): create task NNN <slug>"
```

---

## Stage 2 — Execute

### Goal

Implement the approved task in an isolated worktree / branch while keeping the task file continuously updated.

### Required Steps

1. Before starting implementation, update task status to `in_progress`.
2. Record that status change on main if this is part of your workflow.
3. Create branch and worktree from main, for example:
```bash
   git worktree add ../wt-t015-billing-page-filter -b feat/t015-billing-page-filter main
```
4. Enter the worktree:
```bash
   cd ../wt-t015-billing-page-filter
```
5. Perform all subsequent changes only inside this worktree:
   - code
   - tests
   - docs
   - task file
6. Keep `tasks/NNN-slug.md` updated with timestamped progress notes.
7. After each completed subtask, report immediately:
   - what was completed
   - what files were changed
   - what result was achieved
   - any new risk or discovery
8. After each meaningful phase or rollback-safe checkpoint, remind:
   > You can commit here as a rollback checkpoint.
9. Record key implementation decisions, scope changes, and discoveries.
10. If blocked:
    - set task status to `blocked`
    - document blocker
    - document what was tried
    - document unblock condition
    - stop implementation immediately
11. Implement only the approved scope.
12. Do not archive or mark the task as done in this stage.

---

## Stage 3 — Review

### Goal

Review the full implementation before final fixes or archival.

### Required Steps

1. Review the full diff against `main`.
2. Perform self-review and identify:
   - potential bugs
   - risk points
   - weak test coverage
   - maintainability concerns
   - possible simplifications or optimizations
3. Do not apply fixes yet.
4. Fill the issue table in the task file using this exact format:

| # | 问题描述 | 风险等级 | 影响范围（涉及哪些文件/模块） | 推荐修复方式 |

5. Also summarize:
   - what is already complete
   - what was validated
   - what remains uncertain
6. Explicitly pause after review.
7. Wait for user confirmation before any fix or optimization.
8. Remind:
   > You can commit here as a review checkpoint before discussing fixes.

### Review Output

- Structured issue list
- Validation summary
- No fixes applied yet

---

## Stage 4 — Fix

### Goal

Apply only the fixes explicitly approved after review.

### Required Steps

1. Only fix issues explicitly approved by the user.
2. If the user asks questions about review issues:
   - pause all fixes
   - answer questions first
   - do not change code until discussion is resolved
3. For each approved fix:
   - implement the change
   - report what changed
   - report affected files
   - report expected impact
4. After each approved fix or safe batch of fixes, remind:
   > You can commit here as a rollback checkpoint.
5. Update the task file progress log with each applied fix.
6. After all approved fixes are done, summarize:
   - fixed issues
   - remaining unresolved issues
   - tests/checks still recommended before completion
7. Do not mark the task as done until user confirms the repair phase is complete.

---

## Stage 5 — Done

### Goal

Finalize the task and complete repository-level bookkeeping.

### Preconditions

- implementation is complete
- review has been performed
- approved fixes have been applied
- user confirmed the task can be finalized

### Required Steps

1. Update `tasks/NNN-slug.md` with:
   - final completion summary
   - completion stats
   - final commit reference if available
   - `status: done`
2. Move the task file to `tasks/_archive/`
3. Update `PROGRESS.md`
4. Update `LESSONS.md` if there are reusable lessons
5. Summarize:
   - files changed
   - tests/checks run
   - remaining known limitations
   - suggested final commit / merge message
6. Remind:
   > You can commit here as the final completion checkpoint.
7. Do not do opportunistic refactors or new feature work in this stage.

### Example Archive Command
```bash
mkdir -p tasks/_archive
git mv tasks/NNN-slug.md tasks/_archive/NNN-slug.md
```

---

## Status Definitions

| Status       | Meaning                                          |
|--------------|--------------------------------------------------|
| `pending`    | planned but not started                          |
| `in_progress`| implementation is underway                       |
| `blocked`    | cannot proceed until blocker is resolved         |
| `review`     | implementation finished, under review            |
| `fixing`     | approved fixes are being applied                 |
| `done`       | finalized and archived                           |
| `cancelled`  | explicitly stopped and archived if needed        |

---

## Minimum Required Sections in Every Task File

Every task file must contain at least:

- Metadata
- Goal
- Background / Context
- Scope
- Non-goals
- Approach
- Execution Plan
- Test Plan
- Risks
- Open Questions
- Plan Self-Review
- Approval-Ready Summary
- Progress Log
- Decisions
- Working Notes
- Scope Updates
- Blockers
- Review Summary
- Review Issues
- Fix Log
- Completion Summary
- Completion Stats
- Final Commit
- Final Handoff

---

## Reporting Rules for Agents

- Do not delay progress reporting until the end.
- Do not silently expand scope.
- Do not skip test planning.
- Do not skip review just because tests passed.
- Do not fix review findings before explicit approval.
- Do not archive a task before final confirmation.
- Always keep the task file more up to date than chat summaries.

---

## Recommended Repository-Level Companion Files

### PROGRESS.md

Use this file for project-level task overview:

- active tasks
- blocked tasks
- recently completed tasks
- quick links to task files

### LESSONS.md

Use this file for reusable cross-task lessons:

- recurring pitfalls
- implementation patterns
- testing lessons
- review heuristics
- release / merge lessons
