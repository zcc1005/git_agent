# 配置目录

本目录保存可以提交的默认配置和演示配置。

- `default.yaml`：项目默认参数，后续配置整理阶段建立。
- `demo.yaml`：比赛演示参数，后续演示包阶段建立。
- `local.yaml`：本机覆盖配置，不提交 Git。
- `video_sources.json`：录像文件与 RTSP 固定监控源注册表。

配置中的文件路径必须相对于项目根目录。RTSP 地址不得直接写进 JSON，只能通过 `stream.url_env` 引用环境变量，避免摄像头账号和密码进入 Git。

## schema v2

每个视频源通过 `source_kind` 区分：

- `file`：已经录制完成的有限视频文件。
- `rtsp`：持续、无固定总时长的 RTSP 或 RTSPS 视频流。

两种来源共享：

- `source_id`、`display_name`、`line_id`
- `zones`：分屏或区域 ROI，格式 `[x1, y1, x2, y2]`
- `resolution`
- `manifest_path`、`segments`：可选的历史录像索引

`file` 来源使用：

- `video_path`
- `started_at`
- `duration_seconds`
- `manifest_path` 或 `segments`

`rtsp` 来源使用：

```json
{
  "source_kind": "rtsp",
  "video_path": "",
  "started_at": null,
  "duration_seconds": null,
  "stream": {
    "url_env": "MAIN_MONITOR_RTSP_URL",
    "transport": "tcp",
    "capture_window_seconds": 60,
    "segment_seconds": 60,
    "reconnect_seconds": 5.0,
    "connect_timeout_seconds": 10.0,
    "read_timeout_seconds": 15.0
  }
}
```

`transport` 只能是 `tcp`、`udp` 或 `auto`。RTSP 源不要求静态 `started_at`、分辨率或总时长；这些信息将在后续连接探测与监控任务中动态记录。

schema v1 的录像文件配置仍可加载，读取时默认视为 `source_kind=file`。

## 连接探测

配置好对应环境变量后，可通过封闭 Skill 检查 RTSP 源是否在线：

```python
service.run_skill(
    "probe-video-source",
    arguments={"source_id": "main-monitor"},
)
```

该操作只连接视频源并读取一帧，返回分辨率、FPS、编码、后端和连接延迟；不会运行 YOLO、创建检测历史或触发报警。调用方不能传入 RTSP URL、账号密码、传输协议或超时覆盖值。
