---
name: control-alarm
description: Query, explicitly confirm, or explicitly cancel the current industrial belt alarm and persist the action audit. Use when an operator asks to view alarm state or perform a confirmed alarm-control action by alarm ID, session, or line.
---

# Control Alarm

Invoke `AgentService.run_skill("control-alarm", ...)`.

1. `action` is a closed enum: `query`, `confirm`, or `cancel`. Always emit the canonical enum value.
2. Use `action="query"` for every read-only request, including view, show, get, status, 查看, 查询, 显示, 获取, and 状态. Never emit `view`, `show`, `get`, or `status` as the action.
3. Require explicit operator language before using `confirm` or `cancel`; never infer these actions from ambiguous text.
4. Pass `alarm_id` when the operator names one. Otherwise operate on the latest actionable alarm for the session.
5. Add the operator note to the audit record.
6. Report the resulting alarm status exactly as returned.

Read [references/contract.md](references/contract.md) for safety semantics.
