---
name: control-alarm
description: Query, explicitly confirm, or explicitly cancel the current industrial belt alarm and persist the action audit. Use when an operator asks to view alarm state or perform a confirmed alarm-control action by alarm ID, session, or line.
---

# Control Alarm

Invoke `AgentService.run_skill("control-alarm", ...)`.

1. Use `action="query"` for read-only requests.
2. Require explicit operator language before using `confirm` or `cancel`; never infer these actions from ambiguous text.
3. Pass `alarm_id` when the operator names one. Otherwise operate on the latest actionable alarm for the session.
4. Add the operator note to the audit record.
5. Report the resulting alarm status exactly as returned.

Read [references/contract.md](references/contract.md) for safety semantics.
