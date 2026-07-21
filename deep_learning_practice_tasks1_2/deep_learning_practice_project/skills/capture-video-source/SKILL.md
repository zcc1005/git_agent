---
name: capture-video-source
description: Record a bounded local MP4 clip from a registered RTSP source. Use only when the user explicitly asks to record, capture, or save a live monitor segment. This Skill does not run object detection.
---

# Capture Video Source

Invoke `AgentService.run_skill("capture-video-source", ...)` with a registered `source_id`.

1. Resolve display names and line names through `context.video_sources`; never guess identifiers.
2. Pass `duration_seconds` only when the user specifies a recording length. Otherwise use the registered source default.
3. Never pass an RTSP URL, credentials, output directory, filename, codec, transport, or shell argument.
4. Treat the output as an unanalysed local video. Do not claim that detection or risk assessment has run.
5. Use `video_path` and `started_at` from the result when a later approved workflow needs to process the clip.

Read [references/contract.md](references/contract.md) before planner integration.
