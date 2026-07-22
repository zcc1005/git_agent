# Contract

```python
service.run_skill(
    "detect-video-source",
    session_id="operator",
    arguments={
        "source_id": "main-monitor",
        "duration_seconds": 30,
        "zone_id": "belt-zone-a",
        "parameters": {
            "sample_fps": 4.0,
            "conf": 0.25,
            "known_conf": 0.40
        }
    },
)
```

Required input: `source_id`.

Optional inputs:

- `duration_seconds`: numeric `1..3600`; omitted means the source's configured capture window.
- `zone_id`: registered zone identifier. It is mutually exclusive with `parameters.roi`.
- `parameters`: the same closed parameter object used by `detect-video`, including sampling FPS, confidence thresholds, inference size, NMS, tracking, deduplication, event grouping, and ROI.

The deterministic workflow is:

1. Load the registered RTSP source and optional zone.
2. Run `capture-video-source` and retain its MP4 and safe sidecar metadata.
3. Pass the clip path, capture start/end times, source `line_id`, ROI, and detection parameters to the unchanged `detect-video` adapter.
4. Persist the existing detection and alarm records.
5. Return risk, normalized alarm report, event representative frames, capture evidence, and workflow metadata.

Unknown source or zone identifiers fail before capture. Capture failures never invoke detection. The caller cannot provide a URL, credentials, output path, line override, or arbitrary executable parameter.
