from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from main_pipeline import (
    ALARM_REPORT,
    COMMAND_MEANINGS,
    COMMAND_JSON,
    DEFAULT_ADAPTER_DIR,
    DEFAULT_QWEN_MODEL,
    DEFAULT_SPEECH_CKPT,
    DEFAULT_YOLO_MODEL,
    DETECTION_JSON,
    MIC_COMMAND_WAV,
    PROJECT_ROOT,
    record_microphone,
    run_speech_prediction,
    validate_detection_inputs,
    validate_lora_adapter,
    write_manual_command,
    write_skipped_alarm_report,
)
from task2_yolo.detect_yolo import detect_yiwu, make_skipped_json, read_command, should_start_detection
from task3_alarm.generate_alarm_qwen_lora import generate_alarm_report


app = Flask(__name__)
pipeline_lock = threading.Lock()

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
UPLOAD_DIR = OUTPUTS_DIR / "web_inputs"
ACTIVE_ALARM_REPORT = OUTPUTS_DIR / "alarm_report_active.txt"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def read_text_file(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def save_uploaded_image() -> Path:
    file = request.files.get("image")
    if file is None or not file.filename:
        raise ValueError("请先选择一张需要检测的图片。")

    original_name = secure_filename(file.filename)
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("仅支持 jpg、jpeg、png、bmp、webp 格式图片。")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_path = UPLOAD_DIR / f"{timestamp}_{original_name}"
    file.save(image_path)
    return image_path


def find_visualization_image(image_path: Path) -> Path | None:
    vis_dir = OUTPUTS_DIR / "detections_vis"
    direct_path = vis_dir / image_path.name
    if direct_path.exists():
        return direct_path

    matches = sorted(
        vis_dir.glob(f"*{image_path.stem}*"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def find_latest_visualization_image() -> Path | None:
    vis_dir = OUTPUTS_DIR / "detections_vis"
    if not vis_dir.exists():
        return None
    images = [
        path
        for path in vis_dir.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
    ]
    if not images:
        return None
    return max(images, key=lambda item: item.stat().st_mtime)


def path_to_output_url(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    rel = path.resolve().relative_to(OUTPUTS_DIR.resolve()).as_posix()
    return f"/outputs/{rel}"


def build_response(image_path: Path | None = None) -> Dict[str, Any]:
    command = read_command(COMMAND_JSON)
    detection = read_json_file(DETECTION_JSON)
    alarm_text = read_text_file(ALARM_REPORT)
    vis_image = find_visualization_image(image_path) if image_path else find_latest_visualization_image()

    return {
        "command": command,
        "detection": {
            "status": detection.get("status", "unknown"),
            "num_images": detection.get("num_images", 0),
            "num_detections": detection.get("num_detections", 0),
            "has_yiwu": detection.get("has_yiwu", False),
            "has_foreign_object": detection.get(
                "has_foreign_object", detection.get("has_yiwu", False)
            ),
            "class_counts": detection.get("class_counts", {}),
            "objects": detection.get("objects", []),
            "output": display_path(DETECTION_JSON),
        },
        "alarm_report": alarm_text,
        "paths": {
            "command_json": display_path(COMMAND_JSON),
            "detection_json": display_path(DETECTION_JSON),
            "alarm_report": display_path(ALARM_REPORT),
            "input_image": display_path(image_path) if image_path else "",
            "visualization_image": display_path(vis_image) if vis_image else "",
        },
        "image_url": path_to_output_url(vis_image),
    }


def write_alarm_control_command(command_value: str) -> None:
    command_value = command_value.lower().strip()
    if command_value not in {"yes", "no"}:
        raise ValueError("报警控制命令只能是 yes 或 no。")

    payload = {
        "command": command_value,
        "meaning": COMMAND_MEANINGS[command_value],
        "confidence": 1.0,
        "start_detection": False,
        "confirm_alarm": command_value == "yes",
        "cancel_alarm": command_value == "no",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "web_alarm_control",
    }
    COMMAND_JSON.parent.mkdir(parents=True, exist_ok=True)
    with COMMAND_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_active_alarm_report() -> None:
    if not ALARM_REPORT.exists():
        return
    text = ALARM_REPORT.read_text(encoding="utf-8")
    if "当前报警已停止" in text:
        return
    ACTIVE_ALARM_REPORT.write_text(text, encoding="utf-8")


def restore_active_alarm_report() -> None:
    if not ACTIVE_ALARM_REPORT.exists():
        raise FileNotFoundError("未找到可继续的报警报告，请先运行检测并生成报警。")
    ALARM_REPORT.write_text(ACTIVE_ALARM_REPORT.read_text(encoding="utf-8"), encoding="utf-8")


def write_cancelled_alarm_report() -> None:
    save_active_alarm_report()
    text = (
        "工业皮带异物报警处理记录\n\n"
        "一、报警状态\n"
        "用户在网页端选择 no，当前报警已停止。\n\n"
        "二、控制命令\n"
        "no / 取消报警\n\n"
        "三、处理说明\n"
        "系统保留最近一次检测结果和检测框图，但当前报警提示已取消。"
    )
    ALARM_REPORT.parent.mkdir(parents=True, exist_ok=True)
    ALARM_REPORT.write_text(text + "\n", encoding="utf-8")


@app.get("/")
def index():
    return render_template("web_index.html")


@app.get("/outputs/<path:filename>")
def output_file(filename: str):
    return send_from_directory(OUTPUTS_DIR, filename)


@app.post("/api/mic")
def api_mic():
    try:
        seconds = float(request.form.get("record_seconds", 2.5))
        sample_rate = int(request.form.get("sample_rate", 16000))
        with pipeline_lock:
            record_microphone(MIC_COMMAND_WAV, seconds=seconds, sample_rate=sample_rate)
            run_speech_prediction(PROJECT_ROOT, MIC_COMMAND_WAV, DEFAULT_SPEECH_CKPT, COMMAND_JSON)
            command = read_command(COMMAND_JSON)
            if not command:
                raise ValueError("语音识别完成，但 command.json 为空或读取失败。")
        return jsonify(
            {
                "ok": True,
                "message": "语音命令识别完成。",
                "command": command,
                "wav_path": display_path(MIC_COMMAND_WAV),
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/run")
def api_run():
    try:
        mode = request.form.get("mode", "manual")
        command_value = request.form.get("command", "go")
        conf = float(request.form.get("conf", 0.15))
        top_k = int(request.form.get("top_k", 5))
        qwen_device = request.form.get("qwen_device", "cpu")

        with pipeline_lock:
            image_path = save_uploaded_image()

            if mode == "mic":
                command = read_command(COMMAND_JSON)
                if not command:
                    raise ValueError("请先点击麦克风识别，生成语音命令后再运行检测。")
            else:
                write_manual_command(command_value)
                command = read_command(COMMAND_JSON)

            if not should_start_detection(command):
                make_skipped_json(
                    output_json=DETECTION_JSON,
                    source=image_path,
                    model_path=DEFAULT_YOLO_MODEL,
                    command=command,
                )
                write_skipped_alarm_report(command)
                return jsonify({"ok": True, **build_response(image_path)})

            validate_detection_inputs(image_path, DEFAULT_YOLO_MODEL)
            detect_yiwu(
                source=image_path,
                model_path=DEFAULT_YOLO_MODEL,
                output_json=DETECTION_JSON,
                conf=conf,
            )

            validate_lora_adapter(DEFAULT_ADAPTER_DIR)
            generate_alarm_report(
                detection_json=DETECTION_JSON,
                adapter_dir=DEFAULT_ADAPTER_DIR,
                output_txt=ALARM_REPORT,
                model_name_or_path=DEFAULT_QWEN_MODEL,
                top_k=top_k,
                qwen_device=qwen_device,
            )
            save_active_alarm_report()

            return jsonify({"ok": True, **build_response(image_path)})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/alarm_action")
def api_alarm_action():
    try:
        action = request.form.get("action", "").lower().strip()
        with pipeline_lock:
            if action == "no":
                write_cancelled_alarm_report()
                write_alarm_control_command(action)
                message = "已停止报警。"
            elif action == "yes":
                restore_active_alarm_report()
                write_alarm_control_command(action)
                message = "已确认并继续报警。"
            else:
                raise ValueError("报警控制命令只能是 yes 或 no。")

            return jsonify({"ok": True, "message": message, **build_response()})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


def main() -> None:
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
