# Contract

```python
service.run_skill(
    "capture-video-source",
    session_id="operator",
    arguments={
        "source_id": "main-monitor",
        "duration_seconds": 30,
    },
)
```

Required input: `source_id`.

Optional input: `duration_seconds`, a JSON number from `1` through `3600`. When omitted, the execution layer uses the registered `stream.capture_window_seconds` value.

Unknown fields are rejected. In particular, callers cannot provide `rtsp_url`, `output_path`, `transport`, codecs, or timeout overrides.

Successful output includes `captured=true`, source identity, `started_at`, `ended_at`, encoded duration, frame count, dimensions, FPS, source and output codec, backend, transport, `video_path`, and `metadata_path`. The sidecar JSON contains the same safe metadata.

Failures use safe codes such as `configuration_error`, `connection_timeout`, `connection_failed`, `no_video_frame`, `stream_interrupted`, `writer_failed`, `empty_capture`, or `capture_failed`. Neither results nor sidecars contain the RTSP URL or credentials. Partial files are removed.

This Skill performs a local write but does not run YOLO, create a detection history row, create an alarm, or schedule another capture.
