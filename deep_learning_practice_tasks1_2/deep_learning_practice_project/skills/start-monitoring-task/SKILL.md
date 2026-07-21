---
name: start-monitoring-task
description: Create a bounded, non-continuous background monitoring task for a registered RTSP source. Use when the user explicitly asks to start, schedule, arrange, or run monitoring for a defined time window or duration of no more than 24 hours.
---

# Start Monitoring Task

Invoke `AgentService.run_skill("start-monitoring-task", ...)` only after the user explicitly requests monitoring.

1. Resolve source and zone names only from `context.video_sources`.
2. Require exactly one end condition: `end_time` or `run_duration_seconds`.
3. Use an omitted `start_time` only for an explicit immediate-start request.
4. Copy a deterministic absolute `temporal_resolution` into `start_time` and `end_time`; never reinterpret it.
5. Reject ambiguous schedules, windows over 24 hours, historical requests, unknown sources, and unknown zones.
6. Never invent a source ID, zone, ROI, line ID, URL, credential, or detection observation.

Each round delegates to `detect-video-source`, so existing capture, detection, risk, alarm, representative-frame, and SQLite behavior remains authoritative. Read [references/contract.md](references/contract.md) before planner integration.
