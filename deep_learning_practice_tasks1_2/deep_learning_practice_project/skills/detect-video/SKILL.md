---
name: detect-video
description: Detect industrial belt foreign-object events in video with configurable sampling FPS, confidence thresholds, inference size, NMS, tracking, deduplication, and ROI. Use when a task asks to inspect a video and return object class, location, confidence, occurrence time, representative frames, risk, and alarm identifiers.
---

# Detect Video

Invoke `AgentService.run_skill("detect-video", ...)` with an existing video path.

1. Resolve the video and its real start time before invocation.
2. Convert user sampling language into `sample_fps`; keep the default when unspecified.
3. Convert ROI to full-frame pixel coordinates `[x1, y1, x2, y2]`.
4. Attach `line_id` and the source start time for later time-range and line queries.
5. For a long recording segment, pass `start_offset_seconds` and optional `end_offset_seconds`. The adapter extracts that segment before invoking the unchanged detector.
6. Return event times and representative frames from the detector output. Do not synthesize observations.
7. Use the returned deterministic risk result and alarm state.

Read [references/contract.md](references/contract.md) before passing custom detection parameters.
