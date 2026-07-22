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

Supported `parameters` and defaults: `conf=0.25`, `known_conf=0.40`, `imgsz=800`, `nms_iou=0.40`, `duplicate_iou=0.45`, `duplicate_containment=0.80`, `cross_class_iou=0.70`, `cross_class_containment=0.92`, `max_area_ratio=0.65`, and `confirm_low_confidence_unknown=false`.

Numbers and booleans must use native JSON types, not strings. Require `0 <= conf < known_conf <= 1`, `32 <= imgsz <= 4096`, and overlap ratios in `(0, 1)`. Unknown top-level or nested parameters are rejected before inference.

Important output fields: `detection_id`, `alarm_id`, `risk_level`, `alarm_status`, `detection_count`, `candidate_count`, `class_counts`, `candidate_counts`, `result_json`, `visualization_image`.
