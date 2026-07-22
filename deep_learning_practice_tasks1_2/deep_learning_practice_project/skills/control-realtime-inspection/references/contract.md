# Control realtime inspection contract

## Input

- `action`: `query` or `stop`; default `query`.
- Read aliases `view`, `show`, `status`, and `get` normalize to `query` defensively, but planners must emit `query`.
- `task_id`: optional `realtime-<12 hex>` identifier.
- `source_id`: optional registered source filter.
- `event_id`: optional exact realtime event identifier for a detailed report.
- `latest`: return only the newest confirmed event.
- `active_only`: return only events which are still active.
- `after_event_id`: return only events confirmed after this SQLite cursor.
- `limit`: 1 to 100, default 10.

Use `stop` only for explicit stop language. Without `task_id`, stop only the current session's most recent active realtime inspection. Cross-session access returns `task_not_found`.

## Query output

Return task status, elapsed seconds, frames read/inferred/dropped, actual inference FPS, reconnect and failure counters, aggregated event/alarm counts, highest risk, latest inference/detection/alarm identifiers, latest representative frame, last safe error, and recent aggregated events.

Confirmed events are queryable while the task is still running. Each event includes its stable detection/alarm IDs, active/closed status, class counts, maximum confidence, representative frame, alarm report, alarm status, and bounded AI summary. If no event has reached `min_event_hits`, return `当前实时巡检尚未确认异物事件。` rather than asking the user to wait for task completion.

Statuses are `scheduled`, `connecting`, `running`, `reconnecting`, `stop_requested`, `completed`, `stopped`, `failed`, or `interrupted`.

## Stop behavior

Set `stop_requested`, signal the background task, allow an active batch-1 inference to finish, close active events without creating duplicate detection/alarm records, release the capture/threads/queue, then persist `stopped`.
