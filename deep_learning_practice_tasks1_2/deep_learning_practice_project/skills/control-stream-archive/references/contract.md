# Contract

## Input

```json
{
  "action": "start|stop|query",
  "source_id": "main-monitor",
  "segment_seconds": 60,
  "retention_hours": 24,
  "limit": 100
}
```

- `source_id` is required and must exist in `config/video_sources.json`.
- `segment_seconds` is 1–3600 and is used only by `start`.
- `retention_hours` is 1–720 and is used only by `start`.
- `limit` is 1–1000 and limits `query` segment output.
- Defensive aliases normalize `view/show/status/get` to `query` and `cancel` to `stop`; planners must emit canonical values.

## State

Archive state is one of `stopped`, `starting`, `running`, `stopping`, or `failed`. Segment state is one of `recording`, `ready`, `failed`, or `deleted`.

The recorder continuously saves bounded MP4 files, indexes their real start/end times in SQLite, atomically mirrors the index to `outputs/rtsp_archive/<source_id>/manifest.json`, and removes ready files older than the configured retention period. File deletion is allowed only inside configured recording roots.

After a process restart, active archive states become `failed`; recording never reconnects automatically without an explicit start request.

## Safety

Never expose the resolved RTSP URL, URL environment variable value, username, password, or native FFmpeg exception. Starting and stopping are controlled writes and require explicit user intent.
