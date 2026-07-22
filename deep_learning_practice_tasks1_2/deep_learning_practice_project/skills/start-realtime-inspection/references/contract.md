# Start realtime inspection contract

## Input

- Required: `source_id`.
- Exactly one: timezone-aware ISO 8601 `end_time`, or `run_duration_seconds` from 1 to 86400.
- Optional: timezone-aware future `start_time`; omission means now.
- Optional defaults: `sample_fps=2.0`, `reconnect_interval_seconds=3.0`, `max_consecutive_failures=3`, `min_event_hits=2`, `event_silence_seconds=1.0`.
- `sample_fps` range: 0.2 to 10.
- `parameters`: existing deterministic video detector options (`conf`, `known_conf`, `imgsz`, `nms_iou`, `roi`, duplicate/tracking thresholds).
- `zone_id` resolves to a registered ROI and is mutually exclusive with `parameters.roi`.

Unknown fields are rejected. Never accept `rtsp_url`, `line_id`, `output_path`, `model_path`, shell commands, or Python code.

## Output

On success, return `task_id`, source/zone metadata, start/end time, status, counters, and configuration. Initial status is `scheduled` or `connecting`; execution continues in a bounded background thread.

An event is persisted immediately when it reaches `min_event_hits`; task completion is not required. Continued matching frames update the same event, detection and alarm IDs. After `event_silence_seconds` without a match, the event changes from `active` to `closed`. Task completion closes any remaining active event and produces only a task summary; it does not recreate earlier alarms.

Safe errors include `source_not_found`, `not_rtsp_source`, `zone_not_found`, `invalid_schedule`, `invalid_parameters`, and `gpu_busy`.

## Intent boundaries

- Online probe: `probe-video-source`.
- Capture only: `capture-video-source`.
- Capture then detect one fixed window: `detect-video-source`.
- Reconnecting periodic windows: `start-monitoring-task`.
- Continuously connected sampled inference: this Skill.
- Continuous recording: `control-stream-archive`.
- Past archived range: `detect-archived-video`.
