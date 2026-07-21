# Contract

```python
service.run_skill(
    "start-monitoring-task",
    session_id="operator",
    arguments={
        "source_id": "main-monitor",
        "start_time": "2026-07-21T15:00:00+08:00",
        "end_time": "2026-07-21T16:00:00+08:00",
        "capture_duration_seconds": 30,
        "interval_seconds": 60,
        "zone_id": "belt-zone-a",
        "parameters": {
            "sample_fps": 4.0,
            "conf": 0.25,
            "known_conf": 0.40
        },
        "max_consecutive_failures": 3
    },
)
```

Required input: `source_id`.

End condition: provide exactly one of `end_time` or `run_duration_seconds`. `start_time` and `end_time` must be timezone-aware ISO 8601 values. The scheduled duration must be `1..86400` seconds.

Optional execution inputs:

- `capture_duration_seconds`: per-round RTSP capture window, `1..3600`; omitted uses the registered source default.
- `interval_seconds`: wait after a round finishes before starting the next, `1..86400`, default `60`.
- `zone_id`: registered zone identifier; mutually exclusive with `parameters.roi`.
- `parameters`: closed `detect-video` parameter object.
- `max_consecutive_failures`: `1..10`, default `3`.

The call returns immediately with a `monitor-xxxxxxxxxxxx` task ID. The daemon worker runs bounded rounds in the same application process. A round already executing is not force-killed. Consecutive failures at the configured limit produce `failed`. Application restart changes active tasks to `interrupted` and never resumes them automatically.

State transitions:

```text
scheduled -> running -> completed
scheduled/running -> stop_requested -> stopped
running -> failed
scheduled/running/stop_requested -> interrupted (process restart)
```
