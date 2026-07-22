# Contract

```python
service.run_skill(
    "control-monitoring-task",
    session_id="operator",
    arguments={
        "action": "query",
        "task_id": "monitor-012345abcdef",
        "limit": 10
    },
)
```

Inputs:

- `action`: `query | stop`, default `query`.
- `task_id`: optional `monitor-` ID. Querying it returns task metadata and recent rounds.
- `source_id`: optional registered source filter.
- `limit`: `1..100`, default `10`.

Query without `task_id` lists only the current session's tasks. Query with `task_id` includes recent round status, detection ID, alarm ID, risk, safe error information, and result summary.

Stop without `task_id` selects the most recent active task in the current session. Stop marks `stop_requested`, signals the in-process worker, and becomes `stopped` after the current round or wait exits. Terminal tasks are idempotently returned. A session cannot query or stop another session's task.
