---
name: control-realtime-inspection
description: Query status and aggregated events for a bounded continuous RTSP inspection, or safely request it to stop after the current single-frame inference. Use for “查看/显示/查询实时巡检状态” and explicit “停止实时巡检” requests; do not use to control periodic monitoring or stream archiving.
---

# Control Realtime Inspection

Normalize all read-only requests to `action=query`. Use `action=stop` only when the user explicitly requests stopping. Do not infer a stop action from a status question.

Prefer an explicit `task_id`. When stopping without one, allow the execution layer to select only the current session's most recent active realtime task, optionally filtered by registered `source_id`. Never attempt to stop another session's task.

Read [references/contract.md](references/contract.md) for exact arguments, status fields, and safe error handling.

For live event reads, use `latest`, `active_only`, `after_event_id`, or an exact `event_id`. These reads are valid before the inspection task ends because confirmed events are written to SQLite immediately.
