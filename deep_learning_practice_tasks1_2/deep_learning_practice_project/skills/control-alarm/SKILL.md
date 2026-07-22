---
name: control-alarm
description: Query, explicitly confirm, or explicitly cancel industrial belt alarms and persist action audits. Use for one alarm by ID/session/line, or for an explicit confirm/cancel of every alarm belonging to one realtime inspection task.
---

# Control Alarm

Invoke `AgentService.run_skill("control-alarm", ...)`.

1. `action` is a closed enum: `query`, `confirm`, or `cancel`. Always emit the canonical enum value.
2. Use `action="query"` for every read-only request, including view, show, get, status, 查看, 查询, 显示, 获取, and 状态. Never emit `view`, `show`, `get`, or `status` as the action.
3. Require explicit operator language before using `confirm` or `cancel`; never infer these actions from ambiguous text.
4. Use `scope="single"` for one alarm. Pass `alarm_id` when named; otherwise use the latest actionable session alarm.
5. Use `scope="realtime_task"` only for explicit language such as “确认本轮报警” or “取消本轮报警”. Pass the registered `task_id` when available; otherwise resolve the latest realtime task in the current session.
6. Never combine `scope="realtime_task"` with `alarm_id`.
7. Add the operator note to every affected audit record.
8. Report affected, unchanged, and skipped counts exactly as returned.

Read [references/contract.md](references/contract.md) for safety semantics.
