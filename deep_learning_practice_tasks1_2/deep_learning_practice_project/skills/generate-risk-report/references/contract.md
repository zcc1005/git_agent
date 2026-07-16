# Contract

Optional inputs: `date`, `start_time`, `end_time`, `risk_level`, `line_id`, `source_type`, `review_status`.

Use either `date` or an explicit interval. If both are present, the explicit interval takes precedence.

Output metrics:

- `detection_count`, `alarm_count`
- `risk_counts`
- `alarm_status_counts`
- `review_counts`
- `source_counts`
- `class_counts`
- `report`

Counts come from SQLite history. Risk decisions come from the existing deterministic alarm rule engine.
