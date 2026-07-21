---
name: detect-archived-video
description: Detect foreign objects in previously archived footage from a registered monitor over an absolute timezone-aware interval. Use for requests such as detecting today's 08:00–09:00 main-monitor footage, where real archive coverage must be verified before reusing the existing video detection, risk, alarm, and history pipeline.
---

# Detect Archived Video

Invoke `AgentService.run_skill("detect-archived-video", ...)` only for an already elapsed absolute monitor interval.

- Resolve monitor display names to a registered `source_id`; never invent a source or pass an RTSP URL.
- Pass timezone-aware `start_time` and `end_time` from deterministic temporal resolution.
- Use this Skill for past monitor footage. Use `detect-video-source` for the current live window and `start-monitoring-task` for a future or from-now monitoring task.
- Verify complete ready-segment coverage before detection. Return coverage gaps or missing file IDs unchanged.
- Never substitute a current RTSP frame for unavailable historical footage.
- Apply a registered `zone_id`, or explicit `parameters.roi`, but never both.
- Reuse the existing `detect-video` pipeline for each overlapping segment and clip only first/last segment boundaries.

Read [references/contract.md](references/contract.md) for inputs, coverage rules, outputs, and error codes.
