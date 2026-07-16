---
name: assess-risk
description: Convert an existing image or video detection JSON into the project's unified alarm schema and deterministic risk result. Use when a task asks for a risk level, stop requirement, reason, alarm report, or handling recommendation from detection results without rerunning YOLO.
---

# Assess Risk

Invoke `AgentService.run_skill("assess-risk", ...)` with `detection_json` or an in-memory `detection` object.

1. Prefer the detector-produced JSON unchanged.
2. Set `source_type` to `image`, `video`, or `auto`.
3. Return `overall_risk` and `generated_report` from `task3_alarm`; never let a language model override risk or stop decisions.
4. Preserve the generated JSON and text report paths for audit.

Read [references/contract.md](references/contract.md) for fields.
