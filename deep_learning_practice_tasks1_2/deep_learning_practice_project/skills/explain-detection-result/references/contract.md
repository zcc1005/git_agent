# Runtime contract

## Inputs

All inputs are optional and unknown fields are rejected.

| Field | Type | Rules |
| --- | --- | --- |
| `detection_id` | string | At most 128 characters. Omit to use the current session's latest detection. |
| `question` | string | Original user question, at most 1000 characters. |
| `question_type` | enum | `risk_reason`, `action_advice`, `similar_history`, `target_position`, or `general`; default `general`. |
| `history_limit` | integer | 1 through 50; default 10. Used only for similar-history evidence. |

The execution layer rejects records from another session.

## Outputs

- `found`: whether an authorized detection was loaded.
- `detection_id`: the authoritative detection identifier.
- `question_type`: normalized question type.
- `authoritative_facts`: source, time, class counts, object count, confidence, rule risk, reasons, rule actions, alarm status, positions, and representative frames loaded from SQLite.
- `history_summary`: bounded same-class history evidence for `similar_history`; otherwise empty.
- `ai_analysis`: model explanation or deterministic fallback.
- `analysis_source`: `llm` or `fallback`.
- `quick_questions`: the four supported follow-up prompts.

The explanation is non-authoritative. `authoritative_facts.risk_level` and `authoritative_facts.alarm_status` always win if prose and data ever disagree.
