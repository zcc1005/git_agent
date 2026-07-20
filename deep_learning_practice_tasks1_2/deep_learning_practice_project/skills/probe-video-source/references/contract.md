# Contract

```python
service.run_skill(
    "probe-video-source",
    session_id="operator",
    arguments={"source_id": "main-monitor"},
)
```

Required input: `source_id`.

The value must match `^[a-z0-9][a-z0-9_-]{0,127}$` and must identify a source already registered in `config/video_sources.json`. Unknown fields are rejected. The RTSP URL is resolved only inside the deterministic execution layer from the source's configured environment variable.

Online output includes `source_id`, `display_name`, `line_id`, `online`, `checked_at`, `latency_ms`, `width`, `height`, `fps`, `codec`, `backend`, and `transport`.

Offline output additionally uses one safe error code: `connection_timeout`, `connection_failed`, `no_video_frame`, or `probe_failed`. Configuration failures use `configuration_error`; a registered non-RTSP source uses `not_rtsp_source`.

The output never contains the resolved RTSP URL or credentials. This Skill opens the stream, reads one frame, releases the connection, and does not invoke YOLO, persist a detection, or create an alarm.
