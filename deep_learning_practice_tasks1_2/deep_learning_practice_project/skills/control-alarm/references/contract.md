# Contract

Inputs:

- `action`: `query` (default), `confirm`, or `cancel`.
- `alarm_id`: optional explicit alarm identifier.
- `line_id`: optional filter for queries.
- `session_only`: restrict query to the invoking session when true.
- `note`: optional operator note for `confirm` or `cancel`.

`confirm` changes status to `confirmed`; `cancel` changes it to `cancelled`. Both append an immutable `alarm_actions` audit row. An `inactive` no-risk record cannot be controlled.

Outputs include `found`, `alarm_id`, `detection_id` when querying, `risk_level`, `alarm_status`, `requires_stop`, and report content.
