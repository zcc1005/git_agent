from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


from project_config import (
    PROJECT_ROOT,
    SPEECH_CKPT_PATH,
    YOLO_MODEL_PATH,
    resolve_project_path,
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task2_yolo.detect_yolo import detect_yiwu, make_skipped_json, read_command, should_start_detection
from task3_alarm.alarm_rule_engine import complete_detection_alarm


COMMAND_MEANINGS = {
    "go": "开始检测",
    "stop": "停止检测",
    "yes": "确认报警",
    "no": "取消报警",
}

COMMAND_JSON = PROJECT_ROOT / "outputs" / "command.json"
MIC_COMMAND_WAV = PROJECT_ROOT / "outputs" / "mic_command.wav"
DETECTION_JSON = PROJECT_ROOT / "outputs" / "detection.json"
ALARM_REPORT = PROJECT_ROOT / "outputs" / "alarm_report.txt"
IMAGE_UNIFIED_ALARM = PROJECT_ROOT / "outputs" / "unified_alarm_image.json"
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "yolo_yiwu" / "images" / "test"
DEFAULT_YOLO_MODEL = YOLO_MODEL_PATH
DEFAULT_SPEECH_CKPT = SPEECH_CKPT_PATH


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_manual_command(command: str) -> Path:
    command = command.lower().strip()
    payload = {
        "command": command,
        "meaning": COMMAND_MEANINGS[command],
        "confidence": 1.0,
        "start_detection": command == "go",
        "confirm_alarm": command == "yes",
        "cancel_alarm": command == "no",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "manual",
    }

    COMMAND_JSON.parent.mkdir(parents=True, exist_ok=True)
    with COMMAND_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return COMMAND_JSON


def record_microphone(wav_path: Path, seconds: float = 1.5, sample_rate: int = 16000) -> Path:
    try:
        import sounddevice as sd
        from scipy.io.wavfile import write as write_wav
    except ImportError as exc:
        raise RuntimeError(
            "缺少麦克风录音依赖，请先安装：pip install sounddevice scipy"
        ) from exc

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(seconds * sample_rate)
    if num_samples <= 0:
        raise ValueError("录音时长必须大于 0 秒")

    try:
        audio = sd.rec(num_samples, samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
    except Exception as exc:
        raise RuntimeError(f"麦克风录音失败：{exc}") from exc

    audio_int16 = (audio.clip(-1.0, 1.0) * 32767).astype("int16")
    write_wav(wav_path, sample_rate, audio_int16)
    return wav_path


def run_speech_prediction(
    project_root: Path,
    wav_path: Path,
    speech_ckpt: Path,
    command_json: Path,
) -> Path:
    script_path = project_root / "task1_speech" / "predict_command.py"
    if not script_path.exists():
        raise FileNotFoundError(f"未找到语音识别脚本：{script_path}")
    if not speech_ckpt.exists():
        raise FileNotFoundError(
            f"未找到语音识别模型：{speech_ckpt}\n"
            "请先运行 task1_speech/train_speech_transformer.py"
        )

    command_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script_path),
        "--wav",
        display_path(wav_path),
        "--ckpt",
        display_path(speech_ckpt),
        "--output",
        display_path(command_json),
    ]
    try:
        subprocess.run(cmd, cwd=project_root, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"语音命令识别失败，退出码：{exc.returncode}") from exc
    if not command_json.exists():
        raise FileNotFoundError(f"语音识别未生成 command.json：{command_json}")
    return command_json


def load_existing_command() -> Dict[str, Any]:
    if not COMMAND_JSON.exists():
        raise FileNotFoundError(
            f"未找到已有 command.json：{COMMAND_JSON}\n"
            "请先运行任务一，或直接使用：python main_pipeline.py --command go"
        )
    command = read_command(COMMAND_JSON)
    if not command:
        raise ValueError(f"command.json 为空或读取失败：{COMMAND_JSON}")
    return command


def write_skipped_alarm_report(command: Dict[str, Any]) -> None:
    del command  # 命令详情已保存在 detection.json 中，此处保持旧函数签名兼容网页调用。
    generate_rule_alarm_report()


def generate_rule_alarm_report() -> Dict[str, Any]:
    detection = read_detection_summary(DETECTION_JSON)
    if not detection:
        raise FileNotFoundError(f"未找到有效 detection.json：{DETECTION_JSON}")
    ruled, _ = complete_detection_alarm(
        detection,
        input_json=DETECTION_JSON,
        output_json=IMAGE_UNIFIED_ALARM,
        output_txt=ALARM_REPORT,
        source_type="image",
    )
    print(f"统一报警 JSON 已生成：{display_path(IMAGE_UNIFIED_ALARM)}")
    print(f"alarm_report.txt 已生成：{display_path(ALARM_REPORT)}")
    return ruled


def read_detection_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def print_detection_summary(path: Path) -> None:
    data = read_detection_summary(path)
    print(f"detection.json 已保存：{display_path(path)}")
    print(f"检测可视化图片已保存：{display_path(path.parent / 'detections_vis')}")
    print(f"检测图片数量：{data.get('num_images', 0)}")
    print(f"检测目标数量：{data.get('num_detections', 0)}")
    print(f"是否检测到异物：{data.get('has_yiwu', False)}")
    print(f"异物类型统计：{data.get('class_counts', {})}")


def validate_detection_inputs(source: Path, yolo_model: Path) -> None:
    if not yolo_model.exists():
        raise FileNotFoundError(
            f"未找到 YOLO 模型，请先运行 task2_yolo/train_yolo.py\n模型路径：{yolo_model}"
        )
    if not source.exists():
        raise FileNotFoundError(f"未找到待检测图片或文件夹：{source}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="工业皮带异物检测与智能报警系统一键流程")
    parser.add_argument("--mic", action="store_true", help="使用麦克风语音输入命令")
    parser.add_argument("--record_seconds", type=float, default=1.5, help="麦克风录音时长，默认 1.5 秒")
    parser.add_argument("--sample_rate", type=int, default=16000, help="麦克风采样率，默认 16000")
    parser.add_argument(
        "--speech_ckpt",
        type=Path,
        default=DEFAULT_SPEECH_CKPT,
        help="语音识别模型路径，默认 runs/speech_transformer/best_model.pt",
    )
    parser.add_argument(
        "--command",
        choices=["go", "stop", "yes", "no"],
        default=None,
        help="手动命令，默认在未指定 --use_existing_command 时使用 go",
    )
    parser.add_argument(
        "--use_existing_command",
        action="store_true",
        help="使用已有 outputs/command.json，不重新生成手动命令",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="待检测图片、文件夹或视频路径，默认 data/yolo_yiwu/images/test",
    )
    parser.add_argument(
        "--yolo_model",
        "--model",
        dest="yolo_model",
        type=Path,
        default=DEFAULT_YOLO_MODEL,
        help="YOLO best.pt 路径，默认 runs/yolo/yiwu_yolov8s_4class/weights/best.pt",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="YOLO 最低检测阈值，低于该值忽略",
    )
    parser.add_argument(
        "--known-conf",
        "--known_conf",
        dest="known_conf",
        type=float,
        default=0.40,
        help="已知类别阈值，conf 到该值之间输出 unknown",
    )
    parser.add_argument("--skip_alarm", action="store_true", help="只运行到 detection.json，不生成报警文本")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = resolve_project_path(args.source)
    yolo_model = resolve_project_path(args.yolo_model)
    speech_ckpt = resolve_project_path(args.speech_ckpt)

    print("========== 工业皮带异物检测与智能报警系统 ==========")
    print(f"项目根目录：{PROJECT_ROOT}")

    if args.mic:
        print("\n[任务1] 麦克风语音命令识别")
        print(f"请在 {args.record_seconds:g} 秒内说出命令：go / stop / yes / no")
        record_microphone(MIC_COMMAND_WAV, args.record_seconds, args.sample_rate)
        print(f"录音已保存：{display_path(MIC_COMMAND_WAV)}")
        print("开始识别语音命令...")
        command_path = run_speech_prediction(PROJECT_ROOT, MIC_COMMAND_WAV, speech_ckpt, COMMAND_JSON)
        print(f"command.json 已生成：{display_path(command_path)}")
        command = read_command(COMMAND_JSON)
        if not command:
            raise ValueError(f"command.json 为空或读取失败：{COMMAND_JSON}")
    elif args.use_existing_command:
        print("当前模式：使用已有 command.json")
        command = load_existing_command()
    else:
        command_value = args.command or "go"
        print("当前模式：手动命令模式")
        command_path = write_manual_command(command_value)
        print(f"command.json 已生成：{display_path(command_path)}")
        command = read_command(COMMAND_JSON)

    command_value = str(command.get("command", "")).lower().strip()
    meaning = command.get("meaning") or COMMAND_MEANINGS.get(command_value, "未知命令")
    print(f"当前命令：{command_value} / {meaning}")

    if not should_start_detection(command):
        print("\n[任务2] 当前命令不启动检测，生成 skipped 状态的 detection.json...")
        make_skipped_json(
            output_json=DETECTION_JSON,
            source=source,
            model_path=yolo_model,
            command=command,
        )
        print_detection_summary(DETECTION_JSON)
        if not args.skip_alarm:
            print("\n[任务3] 生成检测未启动说明报告...")
            write_skipped_alarm_report(command)
        print("\n========== 流程完成 ==========")
        return

    print("\n[任务2] 开始执行 YOLO 异物检测...")
    validate_detection_inputs(source, yolo_model)
    DETECTION_JSON.parent.mkdir(parents=True, exist_ok=True)
    detect_yiwu(
        source=source,
        model_path=yolo_model,
        output_json=DETECTION_JSON,
        conf=args.conf,
        known_conf=args.known_conf,
    )
    print_detection_summary(DETECTION_JSON)

    if args.skip_alarm:
        print("\n已启用 --skip_alarm，跳过任务三报警文本生成。")
        print("\n========== 流程完成 ==========")
        return

    print("\n[任务3] 开始执行统一报警转换与确定性规则评估...")
    ruled = generate_rule_alarm_report()
    print(f"总体风险：{ruled['overall_risk']['level']}")
    print("\n========== 流程完成 ==========")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(exc)
        raise SystemExit(1)
