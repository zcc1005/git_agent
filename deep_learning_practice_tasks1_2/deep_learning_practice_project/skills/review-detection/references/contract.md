# Contract

Inputs:

- `detection_id`: optional only for an unambiguous latest-session reference.
- `action`: required closed enum `confirm`, `reject`, `close`, or `reopen`; no write-action alias is accepted.
- `reviewer`, `note`: optional audit text.

Transitions:

- `confirm` -> `confirmed`
- `reject` -> `rejected`
- `close` -> `closed`
- `reopen` -> `unreviewed`

Each invocation updates current state on `detection_runs` and appends an immutable `detection_review_actions` row. It does not change alarm status.

The planner may emit an action only when the operator's current message explicitly requests the corresponding write. Unknown parameters and action values are rejected before execution.
