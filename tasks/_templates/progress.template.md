# Progress Log Template

<!--
说明：
- 这是进度记录写法模板
- 推荐把这些内容写入各任务文件中的 `## Progress Log`
- 每条记录都应带时间戳
- 每完成一个子任务立刻更新，不要等全部完成后再补
-->

## Progress Log

- <YYYY-MM-DD HH:mm> — Started work in <branch/worktree>.
  - Completed: <what started / what was prepared>
  - Files Changed: <none / file list>
  - Result: <current state>
  - New Risks / Discoveries: <none / details>

- <YYYY-MM-DD HH:mm> — Completed subtask: <subtask name>.
  - Completed: <what exactly was finished>
  - Files Changed:
    - <file 1>
    - <file 2>
  - Result: <what was achieved>
  - New Risks / Discoveries: <risk or discovery>

- <YYYY-MM-DD HH:mm> — Reached checkpoint: <checkpoint name>.
  - Completed: <checkpoint summary>
  - Files Changed:
    - <file 1>
  - Result: <state after checkpoint>
  - New Risks / Discoveries: <details>
  - Reminder: You can commit here as a rollback checkpoint.

- <YYYY-MM-DD HH:mm> — Scope clarification recorded.
  - Completed: <what was clarified>
  - Files Changed: <task file only / code files>
  - Result: <scope is now clearer>
  - New Risks / Discoveries: <out-of-scope discovery / follow-up needed>

- <YYYY-MM-DD HH:mm> — Encountered blocker.
  - Completed: <what was attempted before blocking>
  - Files Changed:
    - <file 1>
  - Result: Task status should be set to blocked.
  - New Risks / Discoveries:
    - Blocker: <what blocks progress>
    - Tried:
      - <attempt 1>
      - <attempt 2>
    - Unblock Condition: <what must happen to continue>

---

## Progress Note Writing Rules

<!--
说明：
- 这是约束，不是记录内容
-->

1. 每条记录必须能回答四件事：
   - 完成了什么
   - 改了哪些文件
   - 达成了什么结果
   - 发现了什么新的风险或信息

2. 关键阶段后应显式提醒：
   - You can commit here as a rollback checkpoint.

3. 如果任务被阻塞：
   - 在 task 文件中把 Status 改成 blocked
   - 写明 blocker / tried / unblock condition
   - 立即停止进一步实现

4. 不允许只写笼统描述，例如：
   - “继续开发中”
   - “修了一些问题”
   - “改了几个文件”
   这些都不够可审计。

---

## Example

- 2026-03-07 10:20 — Started work in feat/t015-billing-page-filter.
  - Completed: Reviewed billing page entry, query hook, and request serializer.
  - Files Changed: none
  - Result: Confirmed filter state currently has no URL persistence.
  - New Risks / Discoveries: Existing sorting issue is unrelated and should stay out of scope.

- 2026-03-07 11:05 — Completed subtask: filter UI wiring.
  - Completed: Added filter bar UI and connected local state to billing page.
  - Files Changed:
    - src/pages/billing/index.tsx
    - src/components/billing/filter-bar.tsx
  - Result: Users can update filter inputs from the page UI.
  - New Risks / Discoveries: Need to ensure pagination resets when filters change.
  - Reminder: You can commit here as a rollback checkpoint.
