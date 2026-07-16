# Contract

Inputs:

- `detection_json`: path to detector JSON; or
- `detection`: parsed detector object.
- `source_type`: `auto` by default, otherwise `image` or `video`.

Outputs:

- `source`
- `detection_summary`
- `events` and `event_count`
- `candidates` and `candidate_count`
- `normalized_detection`

Event objects retain available `bbox_xyxy`, position labels, confidence, track IDs, occurrence times, key frames, and representative frames. This Skill calls the unified format converter only. It does not run YOLO, assign final risk, create an alarm, or modify history.
