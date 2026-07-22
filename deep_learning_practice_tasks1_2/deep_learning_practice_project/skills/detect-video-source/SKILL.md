---
name: detect-video-source
description: Run one on-demand RTSP inspection by capturing a bounded clip from a registered source and passing it through the existing video detector, deterministic risk engine, alarm report, history storage, and representative-frame output.
---

# Detect Video Source

Invoke `AgentService.run_skill("detect-video-source", ...)` for a single current RTSP inspection.

1. Resolve source and zone display names only through `context.video_sources`.
2. Use `duration_seconds` for the current capture window; omit it to use the source default.
3. Prefer a registered `zone_id`. The execution layer converts it to the exact ROI. Do not invent coordinates.
4. Pass detection thresholds only under `parameters` with native JSON types.
5. Never pass an RTSP URL, credentials, local output path, line override, or shell argument.
6. Return the existing detector's alarm report and representative frames without synthesizing observations.

This Skill is an on-demand one-shot workflow, not continuous monitoring or a historical archive query. Read [references/contract.md](references/contract.md) before planner integration.
