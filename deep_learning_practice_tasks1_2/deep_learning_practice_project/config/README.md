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

## 定长片段采集

第四阶段提供 `capture-video-source`，把已注册 RTSP 源录制为本地 MP4：

```python
service.run_skill(
    "capture-video-source",
    arguments={"source_id": "main-monitor", "duration_seconds": 30},
)
```

省略 `duration_seconds` 时使用 `stream.capture_window_seconds`。执行层自行生成输出路径和安全 JSON 元数据，异常时清理半成品。该 Skill 只采集视频，不运行 YOLO、不写检测历史、不创建报警，也不启动循环监控。

## RTSP 单次检测

第五阶段提供 `detect-video-source`，将实时采集片段交给现有视频检测链路：

```python
service.run_skill(
    "detect-video-source",
    arguments={
        "source_id": "main-monitor",
        "duration_seconds": 30,
        "zone_id": "belt-zone-a",
        "parameters": {"sample_fps": 4.0, "conf": 0.25, "known_conf": 0.40}
    },
)
```

执行顺序为：注册源与区域解析 → RTSP 定长采集 → 现有 `detect-video` → 风险研判 → 报警报告 → SQLite 历史入库。`zone_id` 与手工 `parameters.roi` 互斥；线路始终来自视频源注册表，模型不能覆盖。该阶段仍是按需单次检测，不包含常驻循环、定时调度或历史录像索引。

## 非全天候监控任务

第六阶段提供有限时段后台任务。它在同一应用进程中循环调用 `detect-video-source`，每轮继续生成检测历史、风险、报警报告和代表帧：

```python
service.run_skill(
    "start-monitoring-task",
    session_id="operator",
    arguments={
        "source_id": "main-monitor",
        "run_duration_seconds": 3600,
        "capture_duration_seconds": 30,
        "interval_seconds": 60,
        "max_consecutive_failures": 3
    },
)
```

也可提供带时区的 `start_time` 与 `end_time` 进行预约；`end_time` 和 `run_duration_seconds` 必须且只能提供一个，总时长最长 24 小时。通过 `control-monitoring-task` 的 `query` 查看任务与轮次，通过 `stop` 请求人工停止。

当前实现边界：

- 后台任务属于当前单进程 Flask 实例，不是 7×24 常驻服务或外部任务队列。
- 停止请求不会强杀正在进行的采集或 YOLO 推理，只阻止下一轮。
- 应用重启后，未结束任务标记为 `interrupted`，不会自动恢复。
- 后台与手工检测在同一进程内串行使用推理锁，避免模型/GPU 并发争用。
- 暂不提供每日周期调度；每个任务都必须有本次明确结束条件。

正式生产部署时，应把任务执行迁移到独立 worker/任务队列，并增加进程级租约、心跳、幂等恢复和监控告警。

## SQLite 监控运行状态

第七阶段在第六阶段任务审计表之外新增两个运行状态表：

- `monitoring_jobs`：以 `task_id` 为主键，记录视频源、统一运行状态、任务时间窗、片段时长、最近处理时间和最近错误。
- `stream_segments`：记录每个逻辑采集片段的路径、起止时间、处理状态、关联检测和重试次数。

`monitoring_jobs.status` 只使用：

```text
pending -> connecting -> running -> completed
                         running -> stopping -> cancelled
                         running -> failed
```

预约任务在开始前为 `pending`；每轮采集前为 `connecting`；正常轮询间隔中为 `running`；人工停止先进入 `stopping`，当前轮结束后进入 `cancelled`。应用重启时未完成任务转为 `failed`，且不会自动恢复。

`stream_segments` 使用 `pending | processing | completed | failed`。`segment_id` 是由 `task_id + source_id + 标准化起止时间` 生成的确定性标识，同时数据库对 `(task_id, source_id, started_at, ended_at)` 和非空 `video_path` 建立唯一约束：

- 已完成片段再次认领时直接跳过，不重复调用检测。
- 正在处理的片段不能被另一个 worker 同时认领。
- 失败片段可重新认领，并原子增加 `retry_count`。
- 进程重启时残留的 `processing` 片段转为 `failed`，为后续显式恢复机制保留状态。

第六阶段的 `monitoring_tasks` 与 `monitoring_task_runs` 继续保存任务配置和逐轮审计；新表保存可供 worker、状态接口和恢复逻辑使用的运行投影。旧数据库启动时会自动为已有监控任务补建 `monitoring_jobs`，无需手工迁移。

## 后台执行与 Web 轮询

第八阶段沿用本地后台线程执行器。创建任务后 Flask 请求立即返回，RTSP 采集和检测在 daemon worker 中继续执行；状态与事件接口只读取 SQLite，不等待当前采集或 YOLO 推理完成。

### 创建任务

```http
POST /api/agent/monitoring/start
Content-Type: application/json

{
  "session_id": "operator-session",
  "source_id": "main-monitor",
  "run_duration_seconds": 600,
  "capture_duration_seconds": 30,
  "interval_seconds": 60,
  "max_consecutive_failures": 3
}
```

也可用带时区的 `start_time + end_time` 预约。接口只接受注册表中的 `source_id/zone_id`，不接受 RTSP URL、账号、密码或输出路径。成功响应包含 `task_id` 和推荐的 `2000ms` 轮询间隔。

### 停止任务

```http
POST /api/agent/monitoring/stop
Content-Type: application/json

{
  "session_id": "operator-session",
  "task_id": "monitor-012345abcdef"
}
```

停止只设置停止信号。正在进行的片段会完成，之后任务进入 `cancelled`。

### 状态与增量事件

```http
GET /api/agent/monitoring/status?session_id=operator-session&task_id=monitor-012345abcdef
GET /api/agent/monitoring/events?session_id=operator-session&task_id=monitor-012345abcdef&after_segment_id=segment-xxx
```

`status` 返回连接状态、当前片段、阶段进度、最新报警、停止原因、轮次计数和时间信息。`events` 返回新增片段事件、最新报警和 `next_cursor`；调用方把 `next_cursor` 作为下一次 `after_segment_id`，即可避免重复消费片段事件。

聊天前端默认每 2 秒轮询 `events`，展示：

- 连接/采集状态。
- 当前片段及处理状态。
- 检测阶段和阶段估算进度；这不是虚构的逐帧 YOLO 百分比。
- 最新报警、风险等级和规范化报告。
- 完成、失败或人工取消原因。

浏览器关闭或普通聊天请求结束不会停止后台任务。当前方案用于本地单进程验证；部署多进程 Flask 或需要跨机器恢复时，再把相同 SQLite/Skill 契约迁移到 RQ、Celery 或独立 worker，并增加租约与心跳。APScheduler 更适合后续周期性调度，不用于替代当前片段执行队列。
