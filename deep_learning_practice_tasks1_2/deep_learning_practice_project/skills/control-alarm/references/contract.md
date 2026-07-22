# Contract

Inputs:

- `action`: closed enum `query` (default), `confirm`, or `cancel`.
  - All read-only meanings such as view/show/get/status and 查看/查询/显示/获取/状态 must be serialized as `query`.
  - The runtime may normalize `view`, `show`, `get`, and `status` to `query` only as a backward-compatible, read-only safeguard. Callers must not rely on those aliases.
  - No alias is accepted for `confirm` or `cancel`; both require explicit operator intent.
- `alarm_id`: optional explicit alarm identifier.
- `scope`: `single` (default) or `realtime_task`.
  - `realtime_task` is accepted only with an explicit `confirm` or `cancel` instruction for the whole realtime inspection round.
- `task_id`: optional realtime inspection task ID. It is used only with `scope="realtime_task"`; if omitted, resolve the latest realtime task owned by the current session.
- `line_id`: optional filter for queries.
- `session_only`: restrict query to the invoking session when true.
- `note`: optional operator note for `confirm` or `cancel`.

`confirm` changes status to `confirmed`; `cancel` changes it to `cancelled`. Both append an immutable `alarm_actions` audit row. An `inactive` no-risk record cannot be controlled.

For `scope="realtime_task"`, update each non-inactive alarm linked through `realtime_inspection_events`. Skip alarms already in the requested target state and keep their audit history unchanged. Reject cross-session task access. Return affected, unchanged, and skipped counts.

Outputs include `found`, `alarm_id`, `detection_id` when querying, `risk_level`, `alarm_status`, `requires_stop`, and report content.
