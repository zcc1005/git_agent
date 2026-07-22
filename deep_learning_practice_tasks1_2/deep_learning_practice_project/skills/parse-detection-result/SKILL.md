---
name: parse-detection-result
description: Normalize an existing industrial belt image or video detection JSON and expose source metadata, confirmed objects, candidate objects, event times, positions, confidence, tracks, and representative frames. Use when a task needs to inspect or explain detector output without rerunning inference or applying risk rules.
---

# Parse Detection Result

Invoke `AgentService.run_skill("parse-detection-result", ...)` with `detection_json` or an in-memory `detection` object.

1. Preserve the detector-produced JSON as evidence.
2. Set `source_type` to `image`, `video`, or `auto`.
3. Read confirmed events from `events` and low-confidence observations from `candidates`.
4. Return positions, confidence, real or relative time, tracks, and representative frames only when present in the source result.
5. Use `assess-risk` separately when a risk level or handling advice is requested.

Read [references/contract.md](references/contract.md) for output fields.
