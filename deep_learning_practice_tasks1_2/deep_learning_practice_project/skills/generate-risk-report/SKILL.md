---
name: generate-risk-report
description: Generate a deterministic operational risk summary for a date or filtered interval, including detection volume, risk levels, alarm states, foreign-object classes, media sources, and review closure. Use when a task asks for a daily report, line report, risk summary, or shift handoff.
---

# Generate Risk Report

Invoke `AgentService.run_skill("generate-risk-report", ...)`.

1. Pass `date` for a complete local calendar day, or pass an explicit ISO `start_time` and `end_time`.
2. Add line, risk, source, or review filters only when requested.
3. Use returned database counts and deterministic fields without estimation.
4. Preserve both structured summary data and rendered report text.
5. Disclose an empty interval rather than filling missing data.

Read [references/contract.md](references/contract.md) for inputs and metrics.
