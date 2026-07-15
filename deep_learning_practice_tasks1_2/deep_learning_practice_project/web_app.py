from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from extract_video_frames import save_uploaded_video
from main_pipeline import (
    ALARM_REPORT,
    COMMAND_MEANINGS,
    COMMAND_JSON,
    DEFAULT_SPEECH_CKPT,
    DEFAULT_YOLO_MODEL,
    DETECTION_JSON,
    IMAGE_UNIFIED_ALARM,
    MIC_COMMAND_WAV,
    PROJECT_ROOT,
    record_microphone,
    run_speech_prediction,
    validate_detection_inputs,
    write_manual_command,
    write_skipped_alarm_report,
)
from task2_yolo.detect_yolo import detect_yiwu, make_skipped_json, read_command, should_start_detection
from task3_alarm.alarm_rule_engine import complete_detection_alarm
from video_detection import (
    DEFAULT_DUPLICATE_IOU,
    DEFAULT_EVENT_SILENCE_SECONDS,
    DEFAULT_IMGSZ,
    DEFAULT_MIN_UNKNOWN_HITS,
    DEFAULT_NMS_IOU,
    DEFAULT_TRACK_CENTER_DISTANCE_RATIO,
    DEFAULT_TRACK_MAX_AGE_SECONDS,
    DEFAULT_UNKNOWN_SINGLE_FRAME_CONF,
    detect_video_foreign_objects,
    parse_roi,
    parse_video_start_time,
)


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024
pipeline_lock = threading.Lock()

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
UPLOAD_DIR = OUTPUTS_DIR / "web_inputs"
ACTIVE_ALARM_REPORT = OUTPUTS_DIR / "alarm_report_active.txt"
VIDEO_DETECTIONS_DIR = OUTPUTS_DIR / "video_detections"
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


def build_video_response(result: Dict[str, Any]) -> Dict[str, Any]:
    events = []
    for event in result.get("events", []):
        event_data = dict(event)
        key_frame = PROJECT_ROOT / str(event_data["key_frame"])
        event_data["key_frame_url"] = path_to_output_url(key_frame)
        key_frames = []
        for raw_key_frame in event_data.get("key_frames", []):
            key_frame_data = dict(raw_key_frame)
            image_path = PROJECT_ROOT / str(key_frame_data.get("image", ""))
            key_frame_data["image_url"] = path_to_output_url(image_path)
            key_frames.append(key_frame_data)
        if not key_frames and event_data["key_frame_url"]:
            key_frames.append(
                {
                    "image": event_data["key_frame"],
                    "image_url": event_data["key_frame_url"],
                    "track_ids": event_data.get("track_ids", []),
                    "object_count": event_data.get("object_count", 0),
                    "class_counts": event_data.get("class_counts", {}),
                }
            )
        event_data["key_frames"] = key_frames
        events.append(event_data)

    return {
        "status": result["status"],
        "video": result["video"],
        "video_start_time": result["video_start_time"],
        "video_end_time": result["video_end_time"],
        "duration_seconds": result["duration_seconds"],
        "source_fps": result["source_fps"],
        "sample_fps": result["sample_fps"],
        "sampled_frames": result["sampled_frames"],
        "positive_frames": result["positive_frames"],
        "candidate_frames": result.get("candidate_frames", 0),
        "raw_detection_frames": result.get("raw_detection_frames", result["positive_frames"]),
        "saved_images": result["saved_images"],
        "num_raw_detection_boxes": result.get(
            "num_raw_detection_boxes", result.get("num_detection_boxes", 0)
        ),
        "num_deduplicated_boxes": result.get(
            "num_deduplicated_boxes", result.get("num_detection_boxes", 0)
        ),
        "num_detection_boxes": result.get("num_detection_boxes", 0),
        "unique_object_count": result.get("unique_object_count", 0),
        "has_foreign_object": result["has_foreign_object"],
        "num_events": result["num_events"],
        "class_counts": result["class_counts"],
        "events": events,
        "result_json": result["result_json"],
        "alarm_result_json": result.get("alarm_result_json", ""),
        "alarm_report_path": result.get("alarm_report_path", ""),
        "overall_risk": result.get("overall_risk", {}),
        "alarm_report": result.get("alarm_report", ""),
        "thresholds": result.get("thresholds", {}),
        "temporal_parameters": result.get("temporal_parameters", {}),
        "inference_parameters": result.get("inference_parameters", {}),
    }


def build_response(image_path: Path | None = None) -> Dict[str, Any]:
    command = read_command(COMMAND_JSON)
    detection = read_json_file(DETECTION_JSON)
    alarm_text = read_text_file(ALARM_REPORT)
    unified_alarm = read_json_file(IMAGE_UNIFIED_ALARM)
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
        "alarm": {
            "overall_risk": unified_alarm.get("overall_risk", {}),
            "generated_report": unified_alarm.get("generated_report", {}),
        },
        "paths": {
            "command_json": display_path(COMMAND_JSON),
            "detection_json": display_path(DETECTION_JSON),
            "alarm_report": display_path(ALARM_REPORT),
            "unified_alarm": display_path(IMAGE_UNIFIED_ALARM),
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

            complete_detection_alarm(
                read_json_file(DETECTION_JSON),
                input_json=DETECTION_JSON,
                output_json=IMAGE_UNIFIED_ALARM,
                output_txt=ALARM_REPORT,
                source_type="image",
            )
            save_active_alarm_report()

            return jsonify({"ok": True, **build_response(image_path)})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/video-detect")
def api_video_detect():
    try:
        uploaded_file = request.files.get("video")
        if uploaded_file is None or not uploaded_file.filename:
            raise ValueError("请先选择需要检测的视频。")

        video_start = parse_video_start_time(request.form.get("video_start_time", ""))
        sample_fps = float(request.form.get("fps", "2"))
        conf = float(request.form.get("conf", "0.15"))
        imgsz = int(request.form.get("imgsz", str(DEFAULT_IMGSZ)))
        nms_iou = float(request.form.get("nms_iou", str(DEFAULT_NMS_IOU)))
        duplicate_iou = float(
            request.form.get("duplicate_iou", str(DEFAULT_DUPLICATE_IOU))
        )
        event_silence_seconds = float(
            request.form.get(
                "event_silence_seconds", str(DEFAULT_EVENT_SILENCE_SECONDS)
            )
        )
        track_max_age_seconds = float(
            request.form.get(
                "track_max_age_seconds", str(DEFAULT_TRACK_MAX_AGE_SECONDS)
            )
        )
        min_unknown_hits = int(
            request.form.get("min_unknown_hits", str(DEFAULT_MIN_UNKNOWN_HITS))
        )
        unknown_single_frame_conf = float(
            request.form.get(
                "unknown_single_frame_conf",
                str(DEFAULT_UNKNOWN_SINGLE_FRAME_CONF),
            )
        )
        track_center_distance_ratio = float(
            request.form.get(
                "track_center_distance_ratio",
                str(DEFAULT_TRACK_CENTER_DISTANCE_RATIO),
            )
        )
        agnostic_nms = request.form.get("agnostic_nms", "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        roi = parse_roi(request.form.get("roi", ""))

        with pipeline_lock:
            video_path = save_uploaded_video(uploaded_file)
            output_dir = VIDEO_DETECTIONS_DIR / video_path.stem
            result = detect_video_foreign_objects(
                video_path=video_path,
                model_path=DEFAULT_YOLO_MODEL,
                output_dir=output_dir,
                video_start=video_start,
                sample_fps=sample_fps,
                conf=conf,
                imgsz=imgsz,
                nms_iou=nms_iou,
                agnostic_nms=agnostic_nms,
                duplicate_iou=duplicate_iou,
                event_silence_seconds=event_silence_seconds,
                track_max_age_seconds=track_max_age_seconds,
                min_unknown_hits=min_unknown_hits,
                unknown_single_frame_conf=unknown_single_frame_conf,
                track_center_distance_ratio=track_center_distance_ratio,
                roi=roi,
            )
            result_json = output_dir / "detection_results.json"
            alarm_result_json = output_dir / "unified_alarm.json"
            alarm_report_path = output_dir / "alarm_report.txt"
            ruled_alarm, alarm_report = complete_detection_alarm(
                result,
                input_json=result_json,
                output_json=alarm_result_json,
                output_txt=alarm_report_path,
                source_type="video",
            )
            result["alarm_result_json"] = display_path(alarm_result_json)
            result["alarm_report_path"] = display_path(alarm_report_path)
            result["overall_risk"] = ruled_alarm["overall_risk"]
            result["alarm_report"] = alarm_report
        return jsonify({"ok": True, "video_detection": build_video_response(result)})
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
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
