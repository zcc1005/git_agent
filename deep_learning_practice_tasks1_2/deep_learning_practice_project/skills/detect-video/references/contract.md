# Contract

```python
service.run_skill("detect-video", session_id="operator", arguments={
    "video_path": "data/line-1-0800.mp4",
    "video_start_time": "2026-07-16T08:00:00+08:00",
    "line_id": "line-1",
    "parameters": {
        "sample_fps": 4.0,
        "conf": 0.25,
        "known_conf": 0.40,
        "roi": [120, 80, 1180, 700]
    }
})
```

Required input: `video_path`.

Optional metadata: `video_start_time`, `source_ended_at`, `line_id`.

Supported `parameters`: `sample_fps`, `conf`, `known_conf`, `imgsz`, `nms_iou`, `agnostic_nms`, `duplicate_iou`, `duplicate_containment`, `event_silence_seconds`, `track_max_age_seconds`, `min_unknown_hits`, `unknown_single_frame_conf`, `track_iou`, `track_center_distance_ratio`, `roi`.

Require `sample_fps > 0`, `0 <= conf < known_conf <= 1`, `imgsz >= 32`, and a non-empty ROI within the actual frame.

Important output fields: `detection_id`, `alarm_id`, `risk_level`, `event_count`, `class_counts`, `source_started_at`, `source_ended_at`, `result_json`, `alarm_json`, `alarm_report_path`. Detailed event positions, confidence, real times, and representative frames remain in `result_json` and persisted detection history.
