---
name: review-detection
description: Record human review and closed-loop handling for a persisted detection by confirming, rejecting, closing, or reopening it with reviewer and note audit data. Use when an operator validates a detected object, marks a false positive, completes handling, or reopens a case.
---

# Review Detection

Invoke `AgentService.run_skill("review-detection", ...)` only after an explicit human decision.

1. Pass the requested `detection_id`; omit it only when the operator clearly refers to the latest detection in the same session.
2. Map valid decisions to `confirm`, `reject`, `close`, or `reopen`.
3. Record `reviewer` and `note` whenever supplied.
4. Keep alarm confirmation separate: use `control-alarm` for alarm state changes.
5. Return the stored review state and audit timestamp.

Read [references/contract.md](references/contract.md) for state transitions.
