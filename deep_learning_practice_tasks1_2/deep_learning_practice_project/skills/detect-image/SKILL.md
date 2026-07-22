---
name: detect-image
description: Detect foreign objects in industrial belt images with configurable confidence, inference size, NMS, duplicate removal, and low-confidence candidate handling. Use when a task asks to inspect an image or photo for stone, plastic, metal, wood, or unknown objects and return locations, confidence, visualizations, risk, and alarm identifiers.
---

# Detect Image

Invoke `AgentService.run_skill("detect-image", ...)` with an existing image path.

1. Resolve the requested image path before invocation.
2. Pass only supported parameters; preserve detector defaults when the user does not specify them.
3. Pass `line_id` and capture time when known so history filters remain useful.
4. Return confirmed objects separately from low-confidence candidates. Never promote a candidate in language-model text.
5. Use the returned deterministic risk and alarm fields; do not recalculate them.

Read [references/contract.md](references/contract.md) for the full input and output contract.
