---
name: query-history
description: Query persisted detection history by time interval, risk level, production line, media source, and human-review status. Use when a task asks for past detections, objects, representative evidence, closed-loop state, or records from a date or time window such as today 08:00-09:00.
---

# Query History

Invoke `AgentService.run_skill("query-history", ...)` with only the filters stated by the user.

1. Convert relative dates and times into timezone-aware ISO timestamps before invocation.
   The deterministic planner context resolves `今天`, `昨天`, `今天上午`, explicit clock ranges, and rolling ranges such as `最近一小时`; use its `start_time` and `end_time` unchanged.
2. Treat the requested interval as overlapping the media source interval; fall back to record creation time when source time is unavailable.
3. Use `line_id`, `risk_level`, `source_type`, and `review_status` independently or together.
4. Request details when object events and representative frames are needed.
5. State that no records matched when the result is empty; do not fabricate coverage.

Read [references/contract.md](references/contract.md) for filter values.
