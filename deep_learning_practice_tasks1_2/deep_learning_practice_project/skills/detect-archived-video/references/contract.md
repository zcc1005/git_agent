# Contract

## Input

```json
{
  "source_id": "main-monitor",
  "start_time": "2026-07-21T08:00:00+08:00",
  "end_time": "2026-07-21T09:00:00+08:00",
  "zone_id": "",
  "parameters": {},
  "coverage_tolerance_seconds": 2
}
```

- `source_id`, `start_time`, and `end_time` are required.
- Both times must include a timezone, `end_time > start_time`, and the end must not be in the future.
- `coverage_tolerance_seconds` is 0–10.
- Detection parameters use the same strict protocol as `detect-video`.

## Resolution

Query `ready` archive segments whose ranges overlap the request. Confirm that their union covers the request within the configured boundary tolerance and that every indexed file exists. For each segment calculate:

```text
start_offset = max(0, requested_start - segment_start)
end_offset = min(segment_duration, requested_end - segment_start)
```

Then invoke the existing video detection chain. Do not change `video_detection.py` or YOLO internals.

## Errors

- `archive_coverage_gap`: return `requested_range`, `covered_ranges`, and `gaps`; run no detection.
- `archive_segment_missing`: return missing segment IDs; run no detection.
- `archive_range_in_future`: direct the caller to live detection or a monitoring task.
- `source_not_found` / `zone_not_found`: require a valid registry mapping and never guess.

Successful output includes aggregate risk/event/class counts, per-segment deterministic detection results, normalized alarm report text, and representative event frames.
