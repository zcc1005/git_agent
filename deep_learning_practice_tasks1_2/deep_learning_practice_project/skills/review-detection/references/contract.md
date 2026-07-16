# Contract

Inputs:

- `detection_id`: optional only for an unambiguous latest-session reference.
- `action`: `confirm`, `reject`, `close`, or `reopen`.
- `reviewer`, `note`: optional audit text.

Transitions:

- `confirm` -> `confirmed`
- `reject` -> `rejected`
- `close` -> `closed`
- `reopen` -> `unreviewed`

Each invocation updates current state on `detection_runs` and appends an immutable `detection_review_actions` row. It does not change alarm status.
