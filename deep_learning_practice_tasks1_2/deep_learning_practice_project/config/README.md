# Configuration

本目录用于保存可提交的默认配置和演示配置。

- `default.yaml`：项目默认参数，后续配置整理阶段建立。
- `demo.yaml`：比赛演示参数，后续演示包阶段建立。
- `local.yaml`：本机覆盖配置，不提交 Git。
- `video_sources.json`：长视频源注册表；允许在视频未到位时保留空路径和空元数据。

配置中的路径必须相对于项目根目录，或通过环境变量覆盖；禁止写入个人电脑的绝对路径。

长视频源的稳定字段包括：`source_id`、`display_name`、`video_path`、`started_at`、`line_id`、`zones`、`manifest_path`、`resolution`、`duration_seconds` 和 `segments`。分屏区域使用像素 ROI `[x1, y1, x2, y2]`；原始片段可内嵌在 `segments`，也可由 `manifest_path` 指向后续清单。
