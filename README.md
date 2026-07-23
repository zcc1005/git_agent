# 工业皮带异物智能巡检与报警系统

本项目面向工业皮带输送场景，使用 YOLO 检测石块、塑料、金属、木块和未知异物，并将图片、本地视频、RTSP 视频流、风险研判、报警闭环、历史查询与大模型智能体整合到同一个 Web 界面中。

系统采用“大模型负责理解与编排，Skill 负责确定性执行”的方式：大模型可以理解自然语言，但不能修改检测类别、数量、置信度、风险等级或报警状态。正式检测结果由 YOLO、风险规则引擎和 SQLite 共同记录。

## 主要能力

- 图片检测：上传图片后输出带框结果、异物位置、类别、数量、置信度和风险报告。
- 本地视频检测：按抽帧频率检测视频，聚合连续帧中的同一异物并保存代表帧。
- RTSP 单次检测：探测视频源、录制定长片段，或采集当前一段视频后检测。
- 周期巡检：周期性执行“采集片段 → 完整视频检测 → 等待 → 再次采集”。
- 持续实时巡检：保持一个 RTSP 连接，按 `sample_fps` 抽帧推理，异物确认后立即报警。
- 历史录像：持续按片段归档 RTSP 视频，再按真实时间范围查找和检测历史片段。
- 风险与报警：确定性规则生成风险等级、风险原因、处置建议和结构化报警报告。
- 报警闭环：支持查询、确认、取消报警，以及检测结果的确认、驳回、关闭和重新打开。
- 智能简析与追问：根据已落库的检测事实生成简短解释，并支持风险原因、处置建议、目标位置和同类历史追问。
- 历史统计与日报：按日期、时间、风险等级和线路查询，生成当日风险报告。
- 项目知识问答：回答系统功能、参数、Skill、RTSP、YOLO、输出位置和使用方法等问题。

## 执行链

```text
用户上传文件或输入自然语言
        ↓
本地规则 / 大模型意图理解与任务编排
        ↓
参数校验严格的 Skill
        ↓
图片、视频或 RTSP 检测
        ↓
多帧事件聚合与确定性风险规则
        ↓
检测历史、报警记录和代表帧写入 SQLite / outputs
        ↓
Web 对话、实时巡检任务卡、事件中心和报警中心展示
```

## 1. 环境准备

推荐环境：

- Windows 10/11
- Python 3.9～3.11
- Anaconda 环境 `dl_practice`
- NVIDIA GPU + CUDA（可选；没有 GPU 时可使用 CPU，但推理速度较慢）
- FFmpeg 与 MediaMTX（仅本地 RTSP 模拟、采集和录像归档需要）

安装 Python 依赖：

```powershell
conda activate dl_practice
python -m pip install -r requirements.txt
```

默认部署权重位置：

```text
runs/yolo/yiwu_yolov8s_4class/weights/best.pt
```

如需使用其他权重，可在启动 Web 前设置：

```powershell
$env:YOLO_MODEL_PATH="models/yolo/best.pt"
```

## 2. 配置大模型

首次配置时复制示例文件：

```powershell
Copy-Item .env.example .env
```

在 `.env` 中分别填写供应商密钥。真实密钥不要写入源码、前端或 Git：

```dotenv
LLM_PROVIDER=deepseek

LLM_DEEPSEEK_API_KEY=你的DeepSeek密钥
LLM_DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
LLM_DEEPSEEK_MODEL=deepseek-v4-pro

LLM_C4AI_API_KEY=你的c4ai密钥
LLM_C4AI_BASE_URL=https://c4ai.ccccltd.cn/api/compatible/v1
LLM_C4AI_MODEL=jiaorong-deepseek-v4-pro

LLM_PLANNER_MODE=hybrid
```

推荐使用 `hybrid`：简单明确的指令由本地规则直接执行，复杂任务才调用大模型。这样可以降低延迟和 Token 消耗。

一键切换到 DeepSeek：

```powershell
scripts\switch_llm_provider.cmd deepseek
```

切回 c4ai：

```powershell
scripts\switch_llm_provider.cmd c4ai
```

切换后必须重启 Web 服务，新配置才会生效。脚本只修改 `LLM_PROVIDER`，不会打印或改写密钥。

如果暂时不配置大模型，确定性的图片/视频检测、报警规则和部分固定口令仍可运行；复杂自然语言理解和智能简析将使用规则降级结果。

## 3. 启动 Web 系统

在项目根目录运行：

```powershell
conda activate dl_practice
python web_app.py
```

浏览器打开：

```text
http://127.0.0.1:5000
```

Web 页面支持：

- 上传并预览图片或 MP4 视频；
- 在同一聊天框发起检测和查看报告；
- 查看持续实时巡检任务卡及实时异物事件；
- 查看检测历史、报警中心、今日风险和待处理事项；
- 确认或取消报警；
- 创建多个独立聊天会话；实时巡检任务不会因切换会话而停止。

## 4. 智能体使用示例

### 图片和本地视频

先上传文件但不发指令时，智能体会提示继续输入操作。只发检测指令但没有附件时，智能体会提示先上传图片或视频。

常用指令：

```text
检测这张图片
检测这段视频，每秒抽取2帧，置信度阈值0.3
检测视频第10分钟到第20分钟
为什么这次是高风险？
有什么处置建议？
解释最近一次异物的位置
查看同类历史
```

### 报警、历史和日报

```text
查看当前报警
确认最近一次报警
取消报警 alarm-xxx
查看今天的高风险报警
查看主监控最近一次异物报告
今天有几次高风险报警？
生成今日风险报告
查询昨天上午主线路的检测记录
```

### RTSP 和实时巡检

```text
主监控在线吗
录制主监控10秒
检测主监控当前10秒
从现在开始每隔一分钟检测一次，持续10分钟
从现在开始持续实时巡检主监控2分钟，每秒检测2帧
查看主监控实时巡检状态
查看本次实时巡检发现的所有异物
查看当前仍在持续的报警
停止主监控实时巡检
```

“开启主监控实时巡检”没有结束条件，系统会要求补充运行时长或结束时间。持续实时巡检单次最长运行24小时，不会自动创建无限运行任务。

## 5. 各种视频能力的区别

| 能力 | 连接方式 | 适用场景 | 是否保存完整视频 |
| --- | --- | --- | --- |
| 本地视频检测 | 读取已上传文件 | 离线文件分析 | 保留用户上传的视频 |
| RTSP 单次检测 | 采集一段后断开 | 检测“当前10秒”等短时画面 | 保存本次采集片段 |
| 周期巡检 | 每轮重新采集 | 间隔抽查，如每分钟检测一次 | 每轮会产生采集片段 |
| 持续实时巡检 | 一个长期 RTSP 连接 | 低延迟连续监测 | 默认不保存正常完整视频 |
| 持续录像归档 | 长期 RTSP 分片 | 保留最近若干小时历史录像 | 保存分片并按保留期清理 |
| 历史录像检测 | 查找已归档片段 | 检测过去的指定时间段 | 读取已归档片段 |

持续实时巡检只保留确认异物事件的代表帧和事件 JSON，不缓存所有正常帧，也不会生成每轮 MP4。需要完整历史视频时，应单独开启录像归档。

RTSP 只能看到连接后的画面。要执行“检测今天上午8点到9点的主监控”，必须在这段时间之前已经开启录像归档，且 SQLite 中存在覆盖该时间段的录像片段。

## 6. 配置 RTSP 视频源

视频源注册表位于：

```text
config/video_sources.json
```

注册表保存监控源名称、线路、区域和连接参数，但不直接保存带账号密码的 RTSP 地址。实际地址通过 `.env` 中的环境变量提供：

```dotenv
MAIN_MONITOR_RTSP_URL=rtsp://127.0.0.1:8554/main-monitor
```

默认注册源：

- 监控源编号：`main-monitor`
- 显示名称：皮带主监控
- 线路：`main-line`
- 时区：`Asia/Shanghai`

如果需要“三号线A区”等业务名称，应先在 `config/video_sources.json` 中注册对应线路和区域。区域可配置 ROI，智能体会把 `zone_id` 映射为确定的检测范围，不会让大模型自行猜测区域坐标。

详细字段说明见 [config/README.md](config/README.md)。

## 7. 使用本地 MP4 模拟 RTSP

本地验证可以使用 MediaMTX 接收 FFmpeg 循环推流。

确认依赖：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\rtsp_simulator\setup.ps1 `
  -PythonPath "C:\Users\你的用户名\anaconda3\envs\dl_practice\python.exe"
```

启动循环推流：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\rtsp_simulator\start.ps1 `
  -VideoPath .\data\monitor\output_long.mp4 `
  -StreamName main-monitor
```

验证视频流：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\rtsp_simulator\verify.ps1 `
  -StreamName main-monitor
```

停止模拟器：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\rtsp_simulator\stop.ps1 `
  -StreamName main-monitor
```

默认地址为 `rtsp://127.0.0.1:8554/main-monitor`。完整说明和常见问题见 [scripts/rtsp_simulator/README.md](scripts/rtsp_simulator/README.md)。

## 8. 检测参数

常用参数：

| 参数 | 含义 |
| --- | --- |
| `sample_fps` | 每秒最多送入 YOLO 的抽样帧数 |
| `conf` | 候选检测的最低置信度 |
| `known_conf` | 已知异物进入正式结果的置信度 |
| `imgsz` | YOLO 推理尺寸 |
| `nms_iou` | NMS 去重阈值 |
| `roi` | 检测区域，通常为 `[x1, y1, x2, y2]` |
| `start_offset_seconds` | 本地视频起始偏移秒数 |
| `end_offset_seconds` | 本地视频结束偏移秒数 |
| `min_event_hits` | 实时事件正式确认所需的连续命中次数 |
| `event_silence_seconds` | 多久未再次命中后关闭活动事件 |
| `reconnect_interval_seconds` | RTSP 断开后的重连等待时间 |
| `max_consecutive_failures` | 连续连接或读取失败的停止阈值 |

实时巡检的 `sample_fps` 支持 `0.2～10`，默认值为 `2.0`。`zone_id` 与手动 `roi` 不能同时提供。

所有自然语言参数在执行前都会经过 Skill 协议校验。用户不能通过聊天参数传入任意 Python、系统命令、模型路径、输出路径或未经注册的 RTSP 地址。

## 9. Skill 能力

| Skill | 作用 |
| --- | --- |
| `detect-image` | 检测图片并生成风险与报警结果 |
| `detect-video` | 检测本地视频并聚合多帧事件 |
| `parse-detection-result` | 读取和规范化检测 JSON |
| `assess-risk` | 使用确定性规则进行风险研判 |
| `generate-risk-report` | 按日期、线路和风险等级生成汇总 |
| `query-history` | 查询检测、报警和处理历史 |
| `control-alarm` | 查询、确认或取消报警 |
| `review-detection` | 确认、驳回、关闭或重新打开检测结果 |
| `explain-detection-result` | 解释最近或指定的检测结果 |
| `probe-video-source` | 探测已注册 RTSP 视频源 |
| `capture-video-source` | 定长采集 RTSP 视频并保存 MP4 |
| `detect-video-source` | 采集当前 RTSP 片段后执行完整检测 |
| `start-monitoring-task` | 启动周期性定长采集检测 |
| `control-monitoring-task` | 查询或停止周期监控任务 |
| `start-realtime-inspection` | 启动持续连接的实时巡检 |
| `control-realtime-inspection` | 查询或停止实时巡检 |
| `control-stream-archive` | 启动、查询或停止持续录像归档 |
| `detect-archived-video` | 按真实时间范围检测历史录像 |
| `run-inspection-task` | 组合执行已有巡检任务 |

每个 Skill 的严格输入、输出和安全限制位于：

```text
skills/<skill-name>/SKILL.md
skills/<skill-name>/references/contract.md
```

## 10. 风险、报警与实时事件

检测结果先经过事件聚合，再由 `task3_alarm` 中的确定性规则引擎生成风险等级。大模型只负责解释已经存在的事实。

实时巡检事件状态：

```text
候选事件
  → 达到 min_event_hits
活动事件（立即保存代表帧、创建检测记录和报警）
  → 持续命中时更新次数、置信度和最佳代表帧
  → 超过 event_silence_seconds 未命中
已结束事件
```

同一目标在连续帧中只创建一个事件和一个报警。事件达到确认条件后会立即出现在实时巡检任务卡、事件中心和报警中心，不需要等待整个巡检任务结束。

实时巡检任务与聊天会话相互独立，以 `task_id` 为唯一标识。切换或新建聊天不会停止后台巡检。服务异常重启后，未完成任务会标记为中断，不会静默恢复。

## 11. 输出与数据位置

| 内容 | 默认位置 |
| --- | --- |
| SQLite 历史、报警和任务状态 | `outputs/agent_history.sqlite3` |
| Web 上传文件 | `outputs/web_inputs/` |
| 智能体上传图片 | `outputs/agent_inputs/images/` |
| 浏览器视频预览和封面 | `outputs/agent_inputs/video_previews/` |
| 图片检测 JSON | `outputs/detection.json` |
| 图片带框结果 | `outputs/detections_vis/` |
| Web 视频检测结果 | `outputs/video_detections/` |
| 智能体视频检测结果 | `outputs/agent_video_detections/` |
| 实时巡检代表帧和事件 JSON | `outputs/realtime_inspections/<source_id>/<task_id>/events/` |
| RTSP 模拟验证输出 | `outputs/rtsp_simulator/` |

不要手动删除正在运行任务使用的输出文件。清理历史录像时应使用系统的保留期机制，避免 SQLite 仍引用已被外部删除的片段。

## 12. YOLO 数据与训练

正式数据集默认位于 `data/yolo_yiwu/`，训练类别为：

```text
stone
plastic
metal
wood
```

未知异物用于独立评测和候选处理，不作为第五个训练类别直接混入四分类训练集。

训练前检查数据：

```powershell
python task2_yolo/check_yolo_dataset.py --data data/yolo_yiwu/data.yaml
```

训练：

```powershell
python task2_yolo/train_yolo.py --model yolov8s.pt --epochs 150 --imgsz 800 --batch 8
```

直接运行图片检测：

```powershell
python task2_yolo/detect_yolo.py `
  --source data/yolo_yiwu/images/test `
  --conf 0.25 `
  --known-conf 0.40 `
  --nms-iou 0.40
```

Web、智能体和实时巡检会复用已加载的 YOLO 模型，不会每帧重新加载权重。实时巡检使用有界最新帧队列；推理落后时丢弃旧帧，避免内存随运行时间持续增长。

## 13. 项目结构

```text
agent/               智能体服务、工具、Skill 路由、知识检索与实时巡检
config/              视频源和可提交的默认配置
data/                输入数据、演示视频和 YOLO 数据集
docs/                设计与接入说明
models/              可选部署权重
outputs/             上传文件、检测结果、代表帧、SQLite 和报警报告
runs/                YOLO 训练产物
scripts/             大模型切换与 RTSP 模拟工具
skills/              Skill 说明和严格参数协议
static/               Web 前端脚本和样式
storage/              SQLite 存储层
task2_yolo/           YOLO 训练与图片检测
task3_alarm/          风险规则和报警报告
templates/            Web 页面模板
tests/                自动化测试
video_detection.py   本地视频检测与事件聚合
web_app.py           Web 服务入口
```

更详细的目录边界见 [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)。

## 14. 测试

运行全部测试：

```powershell
python -m pytest -q
```

只运行大模型配置测试：

```powershell
python -m pytest -q tests/test_llm_api.py
```

只运行实时巡检测试：

```powershell
python -m pytest -q tests/test_realtime_inspection.py
```

测试使用假的读取器和检测器覆盖核心实时巡检逻辑，不要求连接真实 RTSP 或真实 YOLO。真实设备验收仍应额外检查 RTSP 连通性、抽帧速率、代表帧、报警闭环和长时间资源占用。

## 15. 常见问题

### Web 对话仍只能识别固定口令

检查 `.env` 中是否已填写当前 `LLM_PROVIDER` 对应的密钥，并在切换供应商后重启 Web。推荐保持 `LLM_PLANNER_MODE=hybrid`。

### 显示 `connecting` 或 `reconnecting`

确认 RTSP 推流端正在运行、地址环境变量正确、8554端口未被其他程序占用，并先使用“主监控在线吗”进行探测。

### 实时巡检没有立即生成报警

异物必须达到 `min_event_hits` 才会成为正式事件。还应检查置信度、ROI、抽帧频率和事件中心；低置信度候选不会直接触发正式报警。

### 无法检测过去的时间段

RTSP 本身不能读取过去画面。只有提前开启录像归档，并且对应时间片段尚未超过保留期，才能检测过去的录像。

### 视频在网页中无法播放

检测使用原始视频，浏览器展示使用单独生成的 H.264 预览和封面。检查 `outputs/agent_inputs/video_previews/` 是否成功生成文件，并确认 FFmpeg 可用。

### GPU 忙或第二个任务无法启动

为避免显存失控，系统会通过检测锁限制并发 YOLO 推理。等待当前任务结束或先停止不再需要的巡检任务。

## 安全说明

- `.env` 已被 Git 忽略，但仍不要截图、提交或发送真实密钥和带凭据的 RTSP URL。
- 大模型只能调用已注册 Skill，不能执行任意命令或 Python 代码。
- 风险等级、检测事实和报警状态以规则引擎与 SQLite 为准。
- 持续实时巡检不保存正常完整视频；需要录像时必须显式开启归档。
- 停止实时巡检会等待当前单帧推理安全结束，再释放读取器、线程和队列。
