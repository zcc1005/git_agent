# Contract

Inputs:

- `media_path` plus `source_type`; or `image_path`; or `video_path`.
- `line_id`.
- Image time: `captured_at` or source time fields.
- Video time: `video_start_time` and optional `source_ended_at`.
- `parameters`: the selected image or video detector parameters.

The workflow is fixed:

1. Call `detect-image` or `detect-video` behavior.
2. Use the existing deterministic alarm rule engine.
3. Persist detection and source metadata in SQLite.
4. Create an inactive or pending alarm record.

The Skill does not discover surveillance files, clip a long recording, train YOLO, modify weights, or delegate risk decisions to a language model. A future planner must resolve file selection and then call this closed contract.
