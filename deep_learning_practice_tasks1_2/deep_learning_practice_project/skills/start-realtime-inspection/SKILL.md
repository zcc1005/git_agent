---
name: start-realtime-inspection
description: Start one bounded, continuously connected RTSP inspection task that samples frames, runs deterministic YOLO inference, aggregates events, evaluates risk, creates alarms, and persists history. Use for requests such as “持续实时巡检”, “持续连接检测”, or “每秒检测N帧”; do not use for one-off capture, fixed-window detection, periodic reconnecting monitoring, archive recording, or historical playback.
---

# Start Realtime Inspection

Resolve `source_id` and optional `zone_id` only from the registered video-source catalog. Require exactly one explicit end condition: `end_time` or `run_duration_seconds`. If neither is present, ask how long to run or when to end; never substitute a default duration.

Pass `sample_fps` at the top level. Keep detection thresholds and ROI-compatible options in `parameters`. Never accept a raw RTSP URL, line ID, output/model path, command, or code from the user. Never provide both `zone_id` and `parameters.roi`.

Invoke this Skill once. It returns immediately with a task ID while the bounded background task maintains one RTSP connection, drops stale sampled frames when inference lags, and stores only aggregated event evidence.

Read [references/contract.md](references/contract.md) before constructing arguments or interpreting errors.
