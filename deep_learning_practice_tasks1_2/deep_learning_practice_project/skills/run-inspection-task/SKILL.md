---
name: run-inspection-task
description: Orchestrate one industrial belt image or video inspection through detection, deterministic risk assessment, SQLite history persistence, and alarm creation. Use when a task requests an end-to-end inspection rather than an isolated detector or report operation, including future LLM-planned natural-language jobs.
---

# Run Inspection Task

Invoke `AgentService.run_skill("run-inspection-task", ...)` after resolving one media source.

1. Pass `source_type` explicitly when known; otherwise let the Skill infer it from `media_path` extension.
2. For recorded video requests such as `today 08:00-09:00`, resolve the actual video path and real `video_start_time` upstream. Do not claim that history metadata is a media archive.
3. Pass line, time, detector parameters, and ROI to the underlying detector without reinterpretation.
4. Return the complete workflow result, including history and alarm identifiers.
5. Use `review-detection` and `control-alarm` for later human closure steps.

Read [references/contract.md](references/contract.md) for orchestration boundaries.
