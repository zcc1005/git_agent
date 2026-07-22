# 工业皮带异物检测与智能报警系统

当前系统包含两项核心能力：

- 异物检测：基于 YOLOv8 识别 `unknown/stone/plastic/metal/wood`，支持图片和视频。
- 规则报警：将检测结果转换为统一事件 JSON，通过确定性规则生成风险等级、处置动作和报警报告。

旧的 `go/stop/yes/no` 语音关键词模型和 LoRA-Qwen 报告链路已下线。网页同时支持手动检测、报警控制和自然语言智能体入口。

## 1. 环境安装

建议使用 PyCharm + Anaconda 环境 `dl_practice`，Python 版本建议 3.9/3.10/3.11。

```bash
pip install -r requirements.txt
```

模型、数据和输出路径统一在 `project_config.py` 中配置。默认使用项目相对路径，也可以在 PowerShell 中用环境变量临时覆盖：

```powershell
$env:YOLO_MODEL_PATH="models/yolo/best.pt"
```

项目目录边界和后续迁移顺序见：

[docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)

统一约定：`data/` 保存输入数据，`outputs/` 保存运行结果，`runs/` 保存训练产物，`models/` 只保存最终部署权重，回归测试统一放在 `tests/`。

如果 `torch` 安装较慢，建议先根据自己的 CUDA 或 CPU 环境安装 PyTorch，再安装其余依赖。

## 2. YOLO 异物检测

原始图片压缩包位于：

```text
data/raw/yolo_images.zip
```

整理 YOLO 数据集：

```bash
python task2_yolo/prepare_yolo_dataset.py
```

人工标注异物框：

```bash
python task2_yolo/annotate_yiwu.py --split train
python task2_yolo/annotate_yiwu.py --split val
python task2_yolo/annotate_yiwu.py --split test
```

当前训练类别为四类，标注快捷键为：`1=stone`、`2=plastic`、`3=metal`、`4=wood`；标签文件中自动保存为 YOLO 要求的连续 ID `0-3`。如果遇到未知异物，按 `5` 后画框并保存，程序会把整张图片及全部框自动移入 `data/yolo_unknown_eval`，不让未知物体作为背景混入训练集。

旧五分类数据只需执行一次自动迁移，不需要重新画框：

```bash
python task2_yolo/migrate_to_four_classes.py
python task2_yolo/migrate_to_four_classes.py --apply
```

第一条命令只检查，第二条才执行。旧 `unknown(0)` 图片和原标签会移到 `data/yolo_unknown_eval`，不参与四分类训练；其余类别 ID 自动减 1，框坐标保持不变。

训练前检查路径、缺失标签、坐标范围和每个划分的类别分布：

```bash
python task2_yolo/check_yolo_dataset.py --data data/yolo_yiwu/data.yaml
```

训练 YOLO：

```bash
python task2_yolo/train_yolo.py --model yolov8s.pt --epochs 150 --imgsz 800 --batch 8
```

`train_yolo.py` 会自动执行同样的数据检查；存在缺失标签或验证集缺类时会停止，避免产生看似正常但指标无效的权重。

根据任务一的 `command.json` 启动检测：

```bash
python task2_yolo/detect_yolo.py --source data/yolo_yiwu/images/test
```

检测采用确认阈值：低于 `0.25` 当作背景，`0.25-0.40` 作为待确认候选且不触发报警，不低于 `0.40` 输出四个已知类别。图片推理默认使用 `imgsz=800`，并执行 NMS、包含关系去重、强重叠跨类别竞争和大背景框过滤。可通过 `--conf`、`--known-conf`、`--nms-iou` 和 `--duplicate-iou` 调整：

```bash
python task2_yolo/detect_yolo.py --source data/yolo_yiwu/images/test --conf 0.25 --known-conf 0.40 --nms-iou 0.40 --duplicate-iou 0.45
```

如需兼容旧流程、让低置信度候选直接作为 `unknown` 报警，可显式添加 `--confirm-low-confidence-unknown`；默认关闭，避免把低分石块误报为未知异物。

调试时可忽略语音命令直接检测：

```bash
python task2_yolo/detect_yolo.py --source data/yolo_yiwu/images/test --ignore_command
```

输出文件：

```text
outputs/detection.json
outputs/detections_vis/
```

## 3. 统一事件与规则报警

任务三目标：

- 将图片或视频检测结果转换成统一报警结构。
- 对每个事件执行确定性风险分级，整体风险取最高事件等级。
- 输出完整统一 JSON 和可读的文本报警报告。
- 规则报告不依赖 LoRA、千问模型或网络连接。

图片检测结果执行规则评估：

```powershell
python -m task3_alarm.alarm_rule_engine `
  --input outputs/detection.json `
  --source-type image `
  --output-json outputs/unified_alarm_image.json `
  --output-txt outputs/alarm_report.txt
```

视频检测结果执行规则评估：

```powershell
python -m task3_alarm.alarm_rule_engine `
  --input outputs/video_detections/<任务目录>/detection_results.json `
  --source-type video `
  --output-json outputs/video_detections/<任务目录>/unified_alarm.json `
  --output-txt outputs/video_detections/<任务目录>/alarm_report.txt
```

主要输出：

```text
events[].risk：逐事件风险和动作
overall_risk：本次检测的总体风险
generated_report：结论、风险说明和处理建议
alarm_report.txt：确定性规则报告
```

运行全部回归测试：

```powershell
python -m unittest discover -s tests -v
```

## 4. 一键流程

准备好 YOLO 权重后，可以运行图片完整流程：

```powershell
python main_pipeline.py --command go --source data/yolo_yiwu/images/test
```

启动支持图片、视频和智能助手的网页：

```powershell
python web_app.py
```

## 持续实时巡检

`start-realtime-inspection` 与旧的 `start-monitoring-task` 相互独立：前者在任务期内保持一个 RTSP 连接并按 `sample_fps` 抽帧推理；后者仍按“采集定长 MP4 后检测、等待、重新连接”的周期方式运行。

持续实时巡检默认不缓存完整视频，不保存正常帧，也不生成每轮 MP4。系统只保存确认异物聚合事件的一张代表帧和事件 JSON，路径为 `outputs/realtime_inspections/<source_id>/<task_id>/events/`。需要持续保存完整历史录像时使用 `control-stream-archive`；需要检测过去时间范围时使用 `detect-archived-video`。

常用口令：

- `从现在开始持续实时巡检主监控2分钟，每秒检测2帧`
- `查看主监控实时巡检状态`
- `停止主监控实时巡检`

实时巡检必须提供运行时长或结束时间，单任务最长 24 小时。服务重启后，未完成任务会标记为 `interrupted`，不会静默恢复。

浏览器访问 `http://127.0.0.1:5000`。

智能助手支持：

- 检测这张图片。
- 检测这段视频。
- 查询上一轮结果。
- 统计今天的高风险报警次数。
- 生成今日风险报告。
- 确认或取消报警。

对话、检测轮次和报警动作保存在 `outputs/agent_history.sqlite3`。当前默认使用规则优先的混合意图识别；未接入大模型时自动运行纯规则逻辑。

最终链路：

```text
手动检测控制 / 自然语言智能体
↓
detection.json + detections_vis
↓
unified_alarm.json + alarm_report.txt
```
