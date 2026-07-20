# Contract

Inputs:

- `media_path` plus `source_type`; or `image_path`; or `video_path`.
- `line_id`.
- Image time: `captured_at` or source time fields.
- Video time: `video_start_time` and optional `source_ended_at`.
- Long-video segment: optional `start_offset_seconds` and `end_offset_seconds`.
- `parameters`: the selected image or video detector parameters.

The workflow is fixed:

1. Call `detect-image` or `detect-video` behavior.
2. Use the existing deterministic alarm rule engine.
3. Persist detection and source metadata in SQLite.
4. Create an inactive or pending alarm record.

The Skill does not discover surveillance files, train YOLO, modify weights, or delegate risk decisions to a language model. A future planner must resolve file selection; when offsets are present, the wrapper extracts the requested recording segment before invoking the unchanged video detector.
