# Contract

Optional inputs: `date`, `start_time`, `end_time`, `risk_level`, `line_id`, `source_type`, `review_status`.

Use either `date` (`YYYY-MM-DD`) or an explicit ISO interval. Combining `date` with either interval boundary is rejected as ambiguous. `risk_level`, `source_type`, and `review_status` are closed enums identical to `query-history`; `line_id` remains a canonical identifier resolved upstream.

Relative date expressions are resolved upstream with `Asia/Shanghai` and injected as deterministic `start_time` and `end_time`; use them unchanged.

Output metrics:

- `detection_count`, `alarm_count`
- `risk_counts`
- `alarm_status_counts`
- `review_counts`
- `source_counts`
- `class_counts`
- `report`

Counts come from SQLite history. Risk decisions come from the existing deterministic alarm rule engine.
