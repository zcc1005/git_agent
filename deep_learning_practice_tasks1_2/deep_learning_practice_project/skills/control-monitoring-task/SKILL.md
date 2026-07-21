---
name: control-monitoring-task
description: Query or stop bounded RTSP monitoring tasks that belong to the current conversation. Use for requests to view monitoring status or rounds, and only use stop when the user explicitly asks to stop, terminate, end, close, or cancel monitoring.
---

# Control Monitoring Task

Invoke `AgentService.run_skill("control-monitoring-task", ...)` with one canonical action.

- Use `action="query"` for view, show, get, status, list, or query requests.
- Use `action="stop"` only for an explicit stop request.
- Never emit `view`, `show`, `get`, `status`, or `cancel`; aliases are accepted only as defensive normalization.
- Restrict every result and stop operation to the current `session_id`.
- If stop omits `task_id`, target only the current session's most recent active task, optionally filtered by registered `source_id`.
- Do not claim that an in-progress capture or inference was killed; stop takes effect after the current round.

Read [references/contract.md](references/contract.md) for inputs, states, and output behavior.
