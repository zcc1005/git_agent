# Contract

Invoke:

```python
service.run_skill("detect-image", session_id="operator", arguments={
    "image_path": "data/belt.jpg",
    "line_id": "line-1",
    "captured_at": "2026-07-16T08:15:00+08:00",
    "parameters": {"conf": 0.25, "known_conf": 0.40, "imgsz": 800}
})
```

Required input: `image_path`.

Optional metadata: `line_id`, `captured_at`, `source_started_at`, `source_ended_at`.

Supported `parameters`: `conf`, `known_conf`, `imgsz`, `nms_iou`, `duplicate_iou`, `duplicate_containment`, `cross_class_iou`, `cross_class_containment`, `max_area_ratio`, `confirm_low_confidence_unknown`.

Require `0 <= conf < known_conf <= 1`, `imgsz >= 32`, and overlap ratios in `(0, 1)`.

Important output fields: `detection_id`, `alarm_id`, `risk_level`, `alarm_status`, `detection_count`, `candidate_count`, `class_counts`, `candidate_counts`, `result_json`, `visualization_image`.
