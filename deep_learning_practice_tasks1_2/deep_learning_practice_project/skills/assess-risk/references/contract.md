# Contract

Inputs:

- `detection_json`: path to detector JSON; or
- `detection`: parsed detector object.
- `source_type`: `auto` by default, otherwise `image` or `video`.

The two detection inputs are alternatives; provide at least one.

Outputs include `risk_level`, `requires_stop`, `reason`, `recommended_action`, `alarm_document`, `alarm_report`, `alarm_json`, and `alarm_report_path`.

This Skill calls `task3_alarm.alarm_rule_engine.complete_detection_alarm`. It does not run image or video inference.
