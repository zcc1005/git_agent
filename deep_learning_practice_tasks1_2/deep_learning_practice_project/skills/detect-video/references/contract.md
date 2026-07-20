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

Optional long-video segment: `start_offset_seconds` is non-negative and defaults to `0`; `end_offset_seconds` is optional and must be greater than `start_offset_seconds`. The wrapper extracts an MP4 segment and advances the real `video_start_time` by `start_offset_seconds` before calling the existing video detector.

The deterministic temporal resolver converts expressions such as `第10分钟到第20分钟` and `从00:30检测到01:20` into these offset fields before Skill invocation. In two-part clock offsets, `MM:SS` is used; three-part offsets use `HH:MM:SS`.

Supported `parameters` and defaults: `sample_fps=4.0`, `conf=0.25`, `known_conf=0.40`, `imgsz=800`, `nms_iou=0.40`, `agnostic_nms=false`, `duplicate_iou=0.45`, `duplicate_containment=0.80`, `event_silence_seconds=1.0`, `track_max_age_seconds=1.0`, `min_unknown_hits=2`, `unknown_single_frame_conf=0.40`, `track_iou=0.15`, `track_center_distance_ratio=3.0`, and `roi=null`.

Numbers, integers, booleans, and arrays must use native JSON types. Require `0 < sample_fps <= 60`, `0 <= conf < known_conf <= 1`, `conf <= unknown_single_frame_conf <= known_conf`, `32 <= imgsz <= 4096`, positive IoU ratios below `1`, and ROI `[x1, y1, x2, y2]` with non-negative integer coordinates and increasing corners. Unknown parameters are rejected before extraction or inference.

Important output fields: `detection_id`, `alarm_id`, `risk_level`, `event_count`, `class_counts`, `source_started_at`, `source_ended_at`, `result_json`, `alarm_json`, `alarm_report_path`. Detailed event positions, confidence, real times, and representative frames remain in `result_json` and persisted detection history.
