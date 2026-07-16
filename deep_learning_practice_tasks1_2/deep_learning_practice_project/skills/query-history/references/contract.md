# Contract

Optional inputs:

- `start_time`, `end_time`: ISO timestamps.
- `risk_level`: `none`, `low`, `medium`, `high`.
- `line_id`: exact production-line identifier.
- `source_type`: `image` or `video`.
- `review_status`: `unreviewed`, `confirmed`, `rejected`, `closed`.
- `limit`: 1 through 1000; default 100.
- `include_details`: include persisted detection JSON and alarm report; default true.

Outputs contain `count` and `records`. Every record includes source metadata, risk, review state, class counts, and creation time. Detailed records also include the original detector result with object positions, confidence, event times, and representative frames where available.
