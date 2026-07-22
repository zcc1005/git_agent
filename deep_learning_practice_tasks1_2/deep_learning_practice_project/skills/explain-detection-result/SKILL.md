---
name: explain-detection-result
description: Explain a persisted industrial-belt detection without changing its authoritative facts. Use when a user asks why the current or specified detection has its risk level, requests handling advice, asks for similar history, wants target-position or bounding-box interpretation, or otherwise follows up on the latest detection result.
---

# Explain Detection Result

Use the registered `explain-detection-result` runtime Skill. Do not query files directly or infer a result from conversation prose.

## Workflow

1. Pass an explicit `detection_id` when the request supplies one. Otherwise omit it so the execution layer reads only the current session's latest detection.
2. Map the request to one `question_type`: `risk_reason`, `action_advice`, `similar_history`, `target_position`, or `general`.
3. Preserve the user's original wording in `question` when available.
4. Treat `authoritative_facts` in the result as immutable. The model explanation cannot override class, count, confidence, risk level, or alarm status.
5. If `found=false`, ask the user to run an image, video, RTSP, archived-video, or realtime detection first.

Read [references/contract.md](references/contract.md) when validating arguments or consuming the returned evidence.

## Safety

- Never accept an RTSP URL, model path, output path, command, Python code, or raw SQL as an argument.
- Never infer weight, real material composition, physical distance, belt speed, or equipment damage.
- Explain pixel boxes and derived image regions as image evidence, not real-world measurements.
- Keep risk level and alarm status anchored to the rule engine and SQLite record.
