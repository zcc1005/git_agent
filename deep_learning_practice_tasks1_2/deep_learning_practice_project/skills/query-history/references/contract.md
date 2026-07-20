# Contract

Optional inputs:

- `date`: local calendar date in `YYYY-MM-DD`; it expands to the complete local day. Do not combine it with `start_time` or `end_time`.
- `start_time`, `end_time`: ISO timestamps.
- `risk_level`: `none`, `low`, `medium`, `high`.
- `line_id`: exact production-line identifier.
- `source_type`: `image` or `video`.
- `review_status`: `unreviewed`, `confirmed`, `rejected`, `closed`.
- `limit`: 1 through 1000; default 100.
- `include_details`: include persisted detection JSON and alarm report; default true.

Use canonical enum values in calls. The runtime safely normalizes Chinese read-filter aliases such as `高风险 -> high`, `图片 -> image`, and `已闭环 -> closed`. It rejects unknown parameters, invalid enum values, non-integer limits, and reversed intervals.

Before Skill invocation, the deterministic temporal resolver uses `current_date`, `current_time`, and `timezone=Asia/Shanghai` to convert `今天`, `今天上午`, `上午8点到9点`, `最近一小时`, `昨天`, and ISO calendar dates into explicit `start_time` and `end_time`. The model must not recalculate those values.

Outputs contain `count` and `records`. Every record includes source metadata, risk, review state, class counts, and creation time. Detailed records also include the original detector result with object positions, confidence, event times, and representative frames where available.
