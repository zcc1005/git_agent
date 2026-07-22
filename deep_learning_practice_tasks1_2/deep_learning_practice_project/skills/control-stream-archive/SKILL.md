---
name: control-stream-archive
description: Start, stop, or query continuous historical recording for a registered RTSP source. Use when a user explicitly asks to begin or stop saving monitor footage, configure segment length or retention, or inspect archive status and indexed segments.
---

# Control Stream Archive

Invoke `AgentService.run_skill("control-stream-archive", ...)` with a registered `source_id` and one canonical action.

- Use `action="start"` only for an explicit request to begin continuous recording or archive retention.
- Use `action="stop"` only for an explicit request to stop recording; it takes effect after the current segment.
- Use `action="query"` for view, show, get, status, or list requests.
- Never pass an RTSP URL. Resolve the display name through the registered video-source catalog.
- Default to 60-second segments and 24-hour retention unless the user specifies valid values.
- Do not claim that starting archive recording also runs YOLO. Detection is a separate Skill.
- Treat SQLite as the source of truth; the manifest is a safe read-only mirror and must not contain stream credentials.

Read [references/contract.md](references/contract.md) for the exact inputs, states, retention behavior, and errors.
