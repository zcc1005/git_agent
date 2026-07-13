# 深度学习实践：语音指令驱动的工业皮带异物检测与智能报警系统

本项目包含三个任务：

- 任务一：基于 Transformer Encoder 的语音命令识别，输出 `outputs/command.json`。
- 任务二：基于 YOLOv8 的工业皮带五类异物检测，类别为 `unknown/stone/plastic/metal/wood`，输出 `outputs/detection.json` 和 `outputs/detections_vis/`。
- 任务三：基于 LoRA 微调 `Qwen/Qwen2.5-0.5B-Instruct`，读取 `outputs/detection.json`，生成工业报警报告 `outputs/alarm_report.txt`。

## 1. 环境安装

建议使用 PyCharm + Anaconda 环境 `dl_practice`，Python 版本建议 3.9/3.10/3.11。

```bash
pip install -r requirements.txt
```

模型、数据和输出路径统一在 `project_config.py` 中配置。默认使用项目相对路径，也可以在 PowerShell 中用环境变量临时覆盖：

```powershell
$env:YOLO_MODEL_PATH="models/yolo/best.pt"
$env:QWEN_MODEL_NAME="models/Qwen2.5-0.5B-Instruct"
```

任务三额外依赖如下，如果当前环境尚未安装，可以单独执行：

```bash
pip install transformers datasets peft accelerate safetensors sentencepiece protobuf
```

如果 `torch/torchaudio` 安装较慢，建议先根据自己的 CUDA 或 CPU 环境安装 PyTorch，再安装其余依赖。

## 2. 任务一：语音命令识别

Google Speech Commands 对应关系：

| 数据集命令 | 中文含义 | 输出 command |
| --- | --- | --- |
| go | 开始检测 | go |
| stop | 停止检测 | stop |
| yes | 确认报警 | yes |
| no | 取消报警 | no |

训练 Transformer 语音模型：

```bash
python task1_speech/train_speech_transformer.py --epochs 10 --batch_size 64
```

使用 wav 文件预测并输出 `outputs/command.json`：

```bash
python task1_speech/predict_command.py --wav your_audio.wav
```

使用电脑麦克风录音识别：

```bash
python task1_speech/record_and_recognize.py --seconds 1.0
```

## 3. 任务二：YOLO 异物检测

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
```

如果旧验证集的框仍然全是 `unknown(0)`，可只复查这些图片。先按 `1-5` 选择类别，再在已有框内单击鼠标右键即可改类，最后按 `s` 保存：

```bash
python task2_yolo/annotate_yiwu.py --split val --review-unknown
```

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

调试时可忽略语音命令直接检测：

```bash
python task2_yolo/detect_yolo.py --source data/yolo_yiwu/images/test --ignore_command
```

输出文件：

```text
outputs/detection.json
outputs/detections_vis/
```

## 4. 任务三：LoRA-Qwen 工业报警文本生成

任务三目标：

- 读取任务二生成的 `outputs/detection.json`。
- 使用 LoRA 微调后的 `Qwen/Qwen2.5-0.5B-Instruct` 生成工业报警报告。
- 输出 `outputs/alarm_report.txt`。

训练数据位于：

```text
task3_alarm/alarm_train_100.jsonl
```

每条样本字段为：

```text
instruction：任务说明
input：检测 JSON 字符串
output：人工编写的标准报警报告
```

检查训练数据格式、类别分布和风险等级分布：

```bash
python task3_alarm/inspect_alarm_dataset.py
```

LoRA 微调：

```bash
python task3_alarm/train_lora_qwen.py
```

默认训练配置：

```text
基础模型：Qwen/Qwen2.5-0.5B-Instruct
epoch：8
batch size：1
gradient accumulation：8
learning rate：2e-4
max sequence length：1024
LoRA target modules：q_proj, k_proj, v_proj, o_proj
```

Windows + RTX 4050 Laptop GPU 显存有限，如果显存不足，可降低序列长度：

```bash
python task3_alarm/train_lora_qwen.py --max_seq_length 768
python task3_alarm/train_lora_qwen.py --max_seq_length 512
```

LoRA 推理生成报警报告：

```bash
python task3_alarm/generate_alarm_qwen_lora.py
```

原始 Qwen 对比生成：

```bash
python task3_alarm/generate_alarm_base_qwen.py
```

任务三输出文件：

```text
outputs/task3_alarm/qwen_alarm_lora：LoRA 适配器和 tokenizer
outputs/alarm_report.txt：LoRA 模型生成的报警报告
outputs/alarm_report_base_qwen.txt：原始 Qwen 生成的对比报告
```

注意：LoRA adapter 不能单独运行，推理时必须同时加载 Qwen2.5-0.5B-Instruct 基础模型和 `outputs/task3_alarm/qwen_alarm_lora` 适配器。

如果无法联网下载 Qwen 模型，请先联网下载，或把模型提前放到本地路径并用 `--model_name_or_path` 指定：

```bash
$env:QWEN_MODEL_NAME="models/Qwen2.5-0.5B-Instruct"
python task3_alarm/train_lora_qwen.py
python task3_alarm/generate_alarm_qwen_lora.py
```

## 5. 一键流程

如果已经完成任务一模型、任务二模型和任务三 LoRA 训练，可以运行：

```bash
python main_pipeline.py --command go --source data/yolo_yiwu/images/test --run_alarm
```

完整作业流程建议：

```bash
python task1_speech/train_speech_transformer.py --epochs 10 --batch_size 64
python task1_speech/record_and_recognize.py --seconds 1.0
python task2_yolo/prepare_yolo_dataset.py
python task2_yolo/annotate_yiwu.py --split train
python task2_yolo/annotate_yiwu.py --split val
python task2_yolo/train_yolo.py --epochs 50 --imgsz 640 --batch 8
python task2_yolo/detect_yolo.py --source data/yolo_yiwu/images/test
python task3_alarm/inspect_alarm_dataset.py
python task3_alarm/train_lora_qwen.py
python task3_alarm/generate_alarm_qwen_lora.py
```

最终链路：

```text
任务一 command.json
↓
任务二 detection.json + detections_vis
↓
任务三 alarm_report.txt
```
