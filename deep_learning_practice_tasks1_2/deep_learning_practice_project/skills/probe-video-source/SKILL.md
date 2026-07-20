---
name: probe-video-source
description: Probe a registered RTSP video source by opening the stream and reading one frame. Use for online status and connection metadata checks. This Skill does not run object detection.
---

# Probe Video Source

Invoke `AgentService.run_skill("probe-video-source", ...)` with a registered `source_id`.

1. Use only a source identifier from `config/video_sources.json`.
2. Never pass an RTSP URL, username, password, timeout, or transport override.
3. Treat `online=false` as an observed connection state, not as an object-detection result.
4. Do not claim that the belt is safe or clear: this Skill reads one frame only and does not run YOLO.
5. Return the deterministic metadata and safe error code exactly as provided.

Read [references/contract.md](references/contract.md) before integrating a planner or API.
