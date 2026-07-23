from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping

import cv2
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from agent import AgentService, AgentTools
from agent.llm_api import (
    LLMAPIConfig,
    OpenAICompatibleClient,
    OpenAICompatibleDetectionExplainer,
    OpenAICompatibleSkillPlanner,
    load_env_file,
)
from extract_video_frames import save_uploaded_video
from main_pipeline import (
    ALARM_REPORT,
    COMMAND_MEANINGS,
    COMMAND_JSON,
    DEFAULT_YOLO_MODEL,
    DETECTION_JSON,
    IMAGE_UNIFIED_ALARM,
    PROJECT_ROOT,
    validate_detection_inputs,
    write_manual_command,
    write_skipped_alarm_report,
)
from task2_yolo.detect_yolo import detect_yiwu, make_skipped_json, read_command, should_start_detection
from task3_alarm.alarm_rule_engine import complete_detection_alarm
from storage import AlarmRecord, SQLiteHistoryStore
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
AGENT_IMAGE_UPLOAD_DIR = OUTPUTS_DIR / "agent_inputs" / "images"
ACTIVE_ALARM_REPORT = OUTPUTS_DIR / "alarm_report_active.txt"
VIDEO_DETECTIONS_DIR = OUTPUTS_DIR / "video_detections"
AGENT_VIDEO_PREVIEW_DIR = OUTPUTS_DIR / "agent_inputs" / "video_previews"
AGENT_HISTORY_DB = OUTPUTS_DIR / "agent_history.sqlite3"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def browser_video_asset_paths(video_path: Path) -> tuple[Path, Path]:
    return (
        AGENT_VIDEO_PREVIEW_DIR / f"{video_path.stem}_browser_h264_640.mp4",
        AGENT_VIDEO_PREVIEW_DIR / f"{video_path.stem}_poster.jpg",
    )


def create_browser_video_assets(video_path: Path) -> Dict[str, str]:
    """Build an H.264 visual proxy and poster without altering the detection input."""
    AGENT_VIDEO_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    preview_path, poster_path = browser_video_asset_paths(video_path)
    if preview_path.is_file() and poster_path.is_file():
        return {"preview_path": str(preview_path), "poster_path": str(poster_path)}

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("无法读取上传视频，不能生成浏览器预览")
    writer = None
    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        if source_fps <= 0 or source_fps > 240:
            source_fps = 25.0
        target_fps = min(source_fps, 24.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width < 1 or height < 1:
            raise ValueError("上传视频缺少有效画面尺寸")
        scale = min(1.0, 640.0 / width)
        preview_width = max(2, int(round(width * scale)) // 2 * 2)
        preview_height = max(2, int(round(height * scale)) // 2 * 2)

        preview_path.unlink(missing_ok=True)
        writer_attempts = (
            (cv2.CAP_MSMF, "avc1"),
            (cv2.CAP_ANY, "avc1"),
            (cv2.CAP_ANY, "H264"),
        )
        for api_preference, codec in writer_attempts:
            candidate = cv2.VideoWriter(
                str(preview_path),
                api_preference,
                cv2.VideoWriter_fourcc(*codec),
                target_fps,
                (preview_width, preview_height),
            )
            if candidate.isOpened():
                writer = candidate
                break
            candidate.release()
        if writer is None:
            raise ValueError("当前环境缺少 H.264 编码器，不能生成浏览器预览")

        frame_index = 0
        next_output_index = 0.0
        frame_step = source_fps / target_fps
        written_frames = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index == 0 and not cv2.imwrite(str(poster_path), frame):
                raise ValueError("无法生成视频首帧封面")
            if frame_index + 1e-6 >= next_output_index:
                preview_frame = (
                    cv2.resize(
                        frame,
                        (preview_width, preview_height),
                        interpolation=cv2.INTER_AREA,
                    )
                    if (preview_width, preview_height) != (width, height)
                    else frame
                )
                writer.write(preview_frame)
                written_frames += 1
                next_output_index += frame_step
            frame_index += 1
        if written_frames < 1:
            raise ValueError("上传视频没有可用于预览的画面")
    except Exception:
        capture.release()
        if writer is not None:
            writer.release()
            writer = None
        preview_path.unlink(missing_ok=True)
        poster_path.unlink(missing_ok=True)
        raise
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if not preview_path.is_file() or preview_path.stat().st_size < 1:
        poster_path.unlink(missing_ok=True)
        raise ValueError("H.264 浏览器预览生成失败")

    return {"preview_path": str(preview_path), "poster_path": str(poster_path)}


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


def save_uploaded_image_file(file, upload_dir: Path = UPLOAD_DIR) -> Path:
    if file is None or not file.filename:
        raise ValueError("请先选择一张需要检测的图片。")

    safe_name = secure_filename(file.filename)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("仅支持 jpg、jpeg、png、bmp、webp 格式图片。")
    safe_stem = Path(safe_name).stem[:80] or "image"
    original_name = f"{safe_stem}{suffix}"

    upload_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_path = upload_dir / f"{timestamp}_{original_name}"
    file.save(image_path)
    return image_path


def save_uploaded_image() -> Path:
    return save_uploaded_image_file(request.files.get("image"))


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
            "num_raw_detections": detection.get("num_raw_detections", 0),
            "num_detections": detection.get("num_detections", 0),
            "num_candidates": detection.get("num_candidates", 0),
            "num_ignored": detection.get("num_ignored", 0),
            "has_yiwu": detection.get("has_yiwu", False),
            "has_foreign_object": detection.get(
                "has_foreign_object", detection.get("has_yiwu", False)
            ),
            "class_counts": detection.get("class_counts", {}),
            "candidate_counts": detection.get("candidate_counts", {}),
            "objects": detection.get("objects", []),
            "candidate_objects": detection.get("candidate_objects", []),
            "ignored_objects": detection.get("ignored_objects", []),
            "thresholds": detection.get("thresholds", {}),
            "inference_parameters": detection.get("inference_parameters", {}),
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


def apply_agent_alarm_control(action: str, alarm: AlarmRecord) -> None:
    """Bridge conversational alarm actions to the existing control command."""
    del alarm
    if action == "confirm":
        write_alarm_control_command("yes")
    elif action == "cancel":
        write_alarm_control_command("no")
    else:
        raise ValueError(f"不支持的智能体报警动作：{action}")


agent_history_store = SQLiteHistoryStore(AGENT_HISTORY_DB)
agent_tools = AgentTools(
    agent_history_store,
    alarm_control_handler=apply_agent_alarm_control,
)


def create_web_agent_service() -> AgentService:
    """Create the Web agent, enabling LLM planning when it is configured."""
    try:
        load_env_file(PROJECT_ROOT / ".env")
        client = OpenAICompatibleClient(LLMAPIConfig.from_env())
        planner = OpenAICompatibleSkillPlanner(client)
        agent_tools.set_detection_explainer(OpenAICompatibleDetectionExplainer(client))
        service = AgentService(
            agent_history_store,
            tools=agent_tools,
            skill_planner=planner,
            skill_planner_mode=os.getenv("LLM_PLANNER_MODE", "hybrid"),
        )
    except (OSError, ValueError) as exc:
        agent_tools.set_detection_explainer(None)
        app.config["AGENT_LLM_ENABLED"] = False
        app.config["AGENT_LLM_INIT_ERROR"] = str(exc)
        app.logger.warning("Web 智能体未启用大模型规划器：%s", exc)
        return AgentService(agent_history_store, tools=agent_tools)

    app.config["AGENT_LLM_ENABLED"] = True
    app.config.pop("AGENT_LLM_INIT_ERROR", None)
    return service


app.config["AGENT_SERVICE"] = create_web_agent_service()


def get_agent_service() -> AgentService:
    service = app.config.get("AGENT_SERVICE")
    if not isinstance(service, AgentService) and not (
        hasattr(service, "chat") and hasattr(service, "history")
    ):
        raise RuntimeError("AGENT_SERVICE 未正确初始化")
    return service


def get_console_history_store() -> SQLiteHistoryStore:
    """Return the authoritative store used by the dashboard and alarm center."""
    configured = app.config.get("AGENT_HISTORY_STORE")
    return configured if isinstance(configured, SQLiteHistoryStore) else agent_history_store


def _console_risk_reason(alarm: AlarmRecord | None) -> str:
    if alarm is None:
        return ""
    document = alarm.report if isinstance(alarm.report, Mapping) else {}
    events = [item for item in document.get("events") or [] if isinstance(item, Mapping)]
    event_risk = events[0].get("risk") if events and isinstance(events[0].get("risk"), Mapping) else {}
    overall = document.get("overall_risk") if isinstance(document.get("overall_risk"), Mapping) else {}
    return str(event_risk.get("reason") or overall.get("reason") or "")


def _console_source_name(record: Any) -> str:
    source_path = str(record.source_path or "")
    if record.source_type == "realtime" and source_path.startswith("realtime://"):
        source_id = source_path.removeprefix("realtime://").split("/", 1)[0]
        return f"{source_id or '监控源'}实时巡检"
    filename = Path(source_path).name
    if filename:
        return filename
    return {
        "image": "图片检测",
        "video": "视频检测",
        "realtime": "实时巡检",
    }.get(record.source_type, "智能体任务")


def _console_record_payload(
    store: SQLiteHistoryStore,
    record: Any,
    alarms_by_detection: Mapping[str, AlarmRecord] | None = None,
) -> Dict[str, Any]:
    detection = record.summary if isinstance(record.summary, Mapping) else {}
    alarm = (
        alarms_by_detection.get(record.id)
        if alarms_by_detection is not None
        else store.get_alarm_for_detection(record.id)
    )
    if record.source_type in {"video", "realtime"}:
        event_frames = AgentTools.video_event_frames(dict(detection))
        event_count = int(detection.get("num_events") or len(detection.get("events") or []))
    else:
        event_frames = AgentTools.image_event_frames(
            dict(detection), str(detection.get("visualization_image") or "")
        )
        event_count = 1 if bool(
            detection.get("has_foreign_object")
            or int(detection.get("num_detections") or 0) > 0
        ) else 0
    representative_frame = str(event_frames[0].get("key_frame") or "") if event_frames else ""
    source_type = "agent" if record.source_type == "realtime" else record.source_type
    class_counts = dict(detection.get("class_counts") or {})
    return {
        "id": record.id,
        "detectionId": record.id,
        "alarmId": alarm.id if alarm is not None else "",
        "createdAt": record.created_at,
        "sourceType": source_type,
        "sourceName": _console_source_name(record),
        "riskLevel": record.risk_level,
        "eventCount": event_count,
        "classCounts": class_counts,
        "summary": f"检测到 {event_count} 个异物事件",
        "report": alarm.report_text if alarm is not None else record.alarm_report,
        "imageUrl": representative_frame,
        "actionStatus": alarm.status if alarm is not None else "inactive",
        "reason": _console_risk_reason(alarm),
        "lineId": record.line_id,
    }


def _console_snapshot_payload(*, limit: int = 200, target_date: str = "") -> Dict[str, Any]:
    store = get_console_history_store()
    day = (
        datetime.strptime(target_date, "%Y-%m-%d").date()
        if target_date
        else datetime.now().astimezone().date()
    )
    records = store.query_detections(limit=limit)
    alarms_by_detection = store.get_alarms_for_detections([record.id for record in records])
    summary = store.daily_summary(day)
    return {
        "date": day.isoformat(),
        "summary": {
            **summary,
            "alarm_status_counts": dict(summary.get("status_counts") or {}),
        },
        "records": [
            _console_record_payload(store, record, alarms_by_detection)
            for record in records
        ],
    }


def enrich_alarm_report_data(data: Any) -> None:
    """Backfill rule-generated report text for chat messages saved before report rendering."""
    if not isinstance(data, dict):
        return
    detection_id = str(data.get("detection_id") or "")
    if detection_id and not data.get("alarm_report"):
        record = agent_history_store.get_detection(detection_id)
        if record is not None and record.alarm_report:
            data["alarm_report"] = record.alarm_report
    if detection_id and not data.get("event_frames"):
        record = agent_history_store.get_detection(detection_id)
        if record is not None and record.source_type in {"video", "realtime"}:
            data["event_frames"] = AgentTools.video_event_frames(record.summary)
        elif record is not None and record.source_type == "image":
            data["event_frames"] = AgentTools.image_event_frames(
                record.summary, str(data.get("visualization_image") or "")
            )
    steps = data.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if isinstance(step, dict):
                enrich_alarm_report_data(step.get("data"))


def _monitoring_session_id(value: Any) -> str:
    session_id = str(value or "default").strip() or "default"
    if len(session_id) > 128:
        raise ValueError("session_id 不能超过 128 个字符")
    return session_id


def _monitoring_request_body() -> Dict[str, Any]:
    if request.is_json:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return dict(payload)
    payload = request.form.to_dict()
    if "parameters" in payload:
        try:
            payload["parameters"] = json.loads(payload["parameters"])
        except json.JSONDecodeError as exc:
            raise ValueError("parameters 必须是 JSON 对象") from exc
    for name in (
        "run_duration_seconds",
        "capture_duration_seconds",
        "interval_seconds",
        "segment_seconds",
        "retention_hours",
        "sample_fps",
        "reconnect_interval_seconds",
        "event_silence_seconds",
    ):
        if name in payload:
            payload[name] = float(payload[name])
    for name in ("max_consecutive_failures", "min_event_hits"):
        if name in payload:
            payload[name] = int(payload[name])
    return payload


def _monitoring_http_status(result: Mapping[str, Any]) -> int:
    if result.get("ok"):
        return 200
    error_code = str(result.get("error_code") or "")
    if error_code in {
        "task_not_found",
        "source_not_found",
        "zone_not_found",
        "archive_not_found",
    }:
        return 404
    if error_code in {
        "invalid_arguments",
        "invalid_schedule",
        "invalid_parameters",
        "invalid_archive_config",
        "archive_range_in_future",
    }:
        return 400
    return 409


def _query_realtime_detail(
    session_id: str, *, task_id: str = "", source_id: str = "", limit: int = 20,
    event_id: str = "", after_event_id: str = "", latest: bool = False,
    active_only: bool = False, events_only: bool = False,
    task_only: bool = False, compact: bool = False,
) -> Dict[str, Any]:
    service = get_agent_service()
    result = service.run_skill(
        "control-realtime-inspection", session_id=session_id,
        arguments={"action": "query", **({"task_id": task_id} if task_id else {}),
                   **({"source_id": source_id} if source_id else {}),
                   **({"event_id": event_id} if event_id else {}),
                   **({"after_event_id": after_event_id} if after_event_id else {}),
                   "latest": latest, "active_only": active_only,
                   "events_only": events_only, "task_only": task_only,
                   "compact": compact, "limit": limit},
    )
    if not result.get("ok") or task_id:
        return result
    data = result.get("data") if isinstance(result.get("data"), Mapping) else {}
    if isinstance(data.get("task"), Mapping):
        return result
    tasks = data.get("tasks", [])
    if not tasks:
        return result
    latest_id = str(tasks[0].get("task_id") or "")
    return service.run_skill(
        "control-realtime-inspection", session_id=session_id,
        arguments={"action": "query", "task_id": latest_id, "limit": limit,
                   **({"event_id": event_id} if event_id else {}),
                   **({"after_event_id": after_event_id} if after_event_id else {}),
                   "latest": latest, "active_only": active_only,
                   "events_only": events_only, "task_only": task_only,
                   "compact": compact},
    )


def _query_monitoring_detail(
    session_id: str,
    *,
    task_id: str = "",
    source_id: str = "",
    limit: int = 20,
) -> Dict[str, Any]:
    service = get_agent_service()
    result = service.run_skill(
        "control-monitoring-task",
        session_id=session_id,
        arguments={
            "action": "query",
            **({"task_id": task_id} if task_id else {}),
            **({"source_id": source_id} if source_id else {}),
            "limit": limit,
        },
    )
    if not result.get("ok") or task_id:
        return result
    data = result.get("data")
    tasks = data.get("tasks") if isinstance(data, Mapping) else []
    if not isinstance(tasks, list) or not tasks:
        return result
    selected_task_id = str(tasks[0].get("task_id") or "")
    if not selected_task_id:
        return result
    return service.run_skill(
        "control-monitoring-task",
        session_id=session_id,
        arguments={
            "action": "query",
            "task_id": selected_task_id,
            "limit": limit,
        },
    )


def _latest_monitoring_alarm(
    session_id: str,
    task: Mapping[str, Any],
    segments: list[Mapping[str, Any]],
) -> Dict[str, Any]:
    alarm_id = str(task.get("last_alarm_id") or "")
    detection_id = str(task.get("last_detection_id") or "")
    if not detection_id:
        detection_id = next(
            (
                str(segment.get("detection_id") or "")
                for segment in segments
                if segment.get("detection_id")
            ),
            "",
        )
    alarm = (
        agent_history_store.get_alarm(alarm_id)
        if alarm_id
        else agent_history_store.get_alarm_for_detection(detection_id)
        if detection_id
        else None
    )
    if alarm is not None and alarm.session_id != session_id:
        alarm = None
    if alarm is None:
        return (
            {
                "alarm_id": alarm_id,
                "detection_id": detection_id,
                "risk_level": str(task.get("last_risk_level") or ""),
            }
            if alarm_id or detection_id
            else {}
        )
    detection = agent_history_store.get_detection(alarm.detection_id)
    event_frames: list[Dict[str, Any]] = []
    if detection is not None:
        event_frames = (
            AgentTools.video_event_frames(detection.summary)
            if detection.source_type == "video"
            else AgentTools.image_event_frames(detection.summary)
        )
    return {
        "alarm_id": alarm.id,
        "detection_id": alarm.detection_id,
        "risk_level": alarm.risk_level,
        "status": alarm.status,
        "requires_stop": alarm.requires_stop,
        "report_text": alarm.report_text,
        "event_frames": event_frames,
        "created_at": alarm.created_at,
        "updated_at": alarm.updated_at,
    }


def _monitoring_snapshot(
    session_id: str,
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    data = result.get("data")
    if not isinstance(data, Mapping) or not isinstance(data.get("task"), Mapping):
        return {
            "found": False,
            "connection": {"state": "idle", "connected": False, "label": "未监控"},
            "current_segment": {},
            "progress": {"phase": "idle", "estimated_percent": 0},
            "latest_alarm": {},
            "stop_reason": "",
        }
    task = dict(data["task"])
    job = dict(data.get("monitoring_job") or {})
    segments = [
        dict(item) for item in data.get("segments") or [] if isinstance(item, Mapping)
    ]
    runs = [dict(item) for item in data.get("runs") or [] if isinstance(item, Mapping)]
    status = str(job.get("status") or task.get("status") or "pending")
    status_labels = {
        "pending": "等待开始",
        "connecting": "正在连接/采集",
        "running": "监控运行中",
        "stopping": "正在停止",
        "completed": "已完成",
        "failed": "运行失败",
        "cancelled": "已取消",
    }
    current_segment = next(
        (item for item in segments if item.get("status") == "processing"),
        segments[0] if segments else {},
    )
    progress_by_status = {
        "pending": ("waiting", 0),
        "connecting": ("capturing_or_detecting", 50 if current_segment else 10),
        "running": ("waiting_next_segment", 100 if runs else 20),
        "stopping": ("stopping_after_current_segment", 90),
        "completed": ("completed", 100),
        "failed": ("failed", 100),
        "cancelled": ("cancelled", 100),
    }
    phase, estimated_percent = progress_by_status.get(status, (status, 0))
    stop_reason = ""
    if status == "failed":
        stop_reason = str(
            job.get("last_error") or task.get("last_error_message") or "监控任务执行失败"
        )
    elif status == "cancelled":
        stop_reason = "用户请求停止监控"
    elif status == "completed":
        stop_reason = "已到达计划结束时间"
    return {
        "found": True,
        "task_id": str(task.get("task_id") or ""),
        "source_id": str(job.get("source_id") or task.get("source_id") or ""),
        "status": status,
        "connection": {
            "state": status,
            "connected": status == "running",
            "label": status_labels.get(status, status),
            "last_error": str(job.get("last_error") or ""),
        },
        "current_segment": current_segment,
        "progress": {
            "phase": phase,
            "estimated_percent": estimated_percent,
            "runs_completed": int(task.get("runs_completed") or 0),
            "runs_succeeded": int(task.get("runs_succeeded") or 0),
            "runs_failed": int(task.get("runs_failed") or 0),
        },
        "latest_alarm": _latest_monitoring_alarm(session_id, task, segments),
        "stop_reason": stop_reason,
        "started_at": str(job.get("started_at") or task.get("start_time") or ""),
        "ends_at": str(job.get("ends_at") or task.get("end_time") or ""),
        "last_processed_at": str(job.get("last_processed_at") or ""),
        "updated_at": str(job.get("updated_at") or task.get("updated_at") or ""),
        "segments": segments,
        "runs": runs,
    }


@app.get("/")
def index():
    return render_template("web_index.html")


@app.get("/api/console/snapshot")
def api_console_snapshot():
    try:
        limit = int(request.args.get("limit", "200"))
        if not 1 <= limit <= 500:
            raise ValueError("limit 必须在 1 到 500 之间")
        target_date = str(request.args.get("date") or "").strip()
        payload = _console_snapshot_payload(limit=limit, target_date=target_date)
        return jsonify({"ok": True, **payload})
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/console/alarms/action")
def api_console_alarm_action():
    try:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象")
        unknown = sorted(set(payload) - {
            "action", "alarm_ids", "detection_ids", "session_id", "note", "all_pending",
            "reconcile_only",
        })
        if unknown:
            raise ValueError(f"请求包含不支持的字段：{', '.join(unknown)}")
        action = str(payload.get("action") or "").strip().lower()
        action = {"yes": "confirm", "no": "cancel"}.get(action, action)
        if action not in {"confirm", "cancel"}:
            raise ValueError("action 只能是 confirm 或 cancel")
        session_id = _monitoring_session_id(payload.get("session_id"))
        alarm_ids = payload.get("alarm_ids") or []
        detection_ids = payload.get("detection_ids") or []
        if not isinstance(alarm_ids, list) or not isinstance(detection_ids, list):
            raise ValueError("alarm_ids 和 detection_ids 必须是数组")
        if len(alarm_ids) + len(detection_ids) > 500:
            raise ValueError("单次最多处理 500 条报警")

        store = get_console_history_store()
        resolved_ids = [str(value).strip() for value in alarm_ids if str(value).strip()]
        if bool(payload.get("all_pending")):
            resolved_ids.extend(alarm.id for alarm in store.list_alarms(status="pending", limit=500))
        detection_alarm_map = store.get_alarms_for_detections(
            [str(detection_id).strip() for detection_id in detection_ids]
        )
        for detection_id in detection_ids:
            alarm = detection_alarm_map.get(str(detection_id).strip())
            if alarm is not None:
                resolved_ids.append(alarm.id)
        resolved_ids = list(dict.fromkeys(resolved_ids))
        if not resolved_ids:
            raise ValueError("没有提供可处理的报警")

        pending_by_id = {
            alarm.id: alarm for alarm in store.list_alarms(status="pending", limit=500)
        }
        pending = [pending_by_id[alarm_id] for alarm_id in resolved_ids if alarm_id in pending_by_id]
        if pending and not bool(payload.get("reconcile_only")):
            handler = app.config.get("ALARM_CONTROL_HANDLER", apply_agent_alarm_control)
            if handler is not None:
                handler(action, pending[0])
        result = store.set_pending_alarm_actions(
            resolved_ids,
            session_id,
            action,
            note=str(payload.get("note") or "网页报警中心批量处置"),
        )
        updated_detections = [
            pending_by_id[alarm_id].detection_id
            for alarm_id in result["updated"]
            if alarm_id in pending_by_id
        ]
        action_text = "确认" if action == "confirm" else "取消"
        return jsonify({
            "ok": True,
            "message": f"已{action_text}{len(result['updated'])}条报警。",
            "action": action,
            "status": "confirmed" if action == "confirm" else "cancelled",
            "affected_count": len(result["updated"]),
            "unchanged_count": len(result["unchanged"]),
            "missing_count": len(result["missing"]),
            "alarm_ids": result["updated"],
            "detection_ids": updated_detections,
            "snapshot": _console_snapshot_payload(limit=200),
        })
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.get("/outputs/<path:filename>")
def output_file(filename: str):
    return send_from_directory(OUTPUTS_DIR, filename)


@app.post("/api/agent/chat")
def api_agent_chat():
    try:
        message = request.form.get("message", "").strip()
        session_id = request.form.get("session_id", "default").strip() or "default"
        uploaded_media = (
            request.files.get("media")
            or request.files.get("image")
            or request.files.get("video")
        )
        has_media = uploaded_media is not None and bool(uploaded_media.filename)
        if not message and not has_media:
            raise ValueError("请输入指令或发送需要分析的图片/视频")
        if len(message) > 4000:
            raise ValueError("聊天消息不能超过 4000 个字符")
        if len(session_id) > 128:
            raise ValueError("session_id 不能超过 128 个字符")

        context: Dict[str, Any] = {}
        with pipeline_lock:
            media_type = ""
            original_name = ""
            if has_media:
                original_name = Path(str(uploaded_media.filename)).name[:255]
                suffix = Path(uploaded_media.filename).suffix.lower()
                if suffix in ALLOWED_EXTENSIONS:
                    media_type = "image"
                    context["image_path"] = str(
                        save_uploaded_image_file(uploaded_media, AGENT_IMAGE_UPLOAD_DIR)
                    )
                else:
                    media_type = "video"
                    video_path = save_uploaded_video(uploaded_media)
                    context["video_path"] = str(video_path)
                    context["_attachment_preview"] = create_browser_video_assets(video_path)
            video_start_time = request.form.get("video_start_time", "").strip()
            if video_start_time:
                context["video_start_time"] = video_start_time
            alarm_id = request.form.get("alarm_id", "").strip()
            if alarm_id:
                context["alarm_id"] = alarm_id
            detection_id = request.form.get("detection_id", "").strip()
            if len(detection_id) > 128:
                raise ValueError("detection_id 不能超过 128 个字符")
            if detection_id:
                context["detection_id"] = detection_id
            task_id = request.form.get("task_id", "").strip()
            if len(task_id) > 128:
                raise ValueError("task_id 不能超过 128 个字符")
            if task_id:
                context["task_id"] = task_id
            task_session_id = request.form.get("task_session_id", "").strip()
            if len(task_session_id) > 128:
                raise ValueError("task_session_id 不能超过 128 个字符")
            if task_session_id and task_id:
                context["task_session_id"] = task_session_id
            if not message:
                response = get_agent_service().receive_attachment(
                    media_type,
                    context[f"{media_type}_path"],
                    session_id=session_id,
                    original_name=original_name,
                    context=context,
                )
                return jsonify(response)
            response = get_agent_service().chat(
                message,
                session_id=session_id,
                context=context,
            )
            if has_media:
                response["attachment_received"] = True
        return jsonify(response)
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.get("/api/agent/history")
def api_agent_history():
    try:
        session_id = request.args.get("session_id", "default").strip() or "default"
        if len(session_id) > 128:
            raise ValueError("session_id 不能超过 128 个字符")
        limit = int(request.args.get("limit", "50"))
        if not 1 <= limit <= 200:
            raise ValueError("limit 必须在 1 到 200 之间")
        messages = get_agent_service().history(session_id, limit=limit)
        for item in messages:
            metadata = item.get("metadata")
            if not isinstance(metadata, dict):
                continue
            enrich_alarm_report_data(metadata.get("data"))
            attachment = metadata.get("attachment")
            if not isinstance(attachment, dict) or attachment.get("media_type") != "video":
                continue
            video_path = Path(str(attachment.get("path") or ""))
            preview_path, poster_path = browser_video_asset_paths(video_path)
            if preview_path.is_file() and poster_path.is_file():
                attachment.setdefault("preview_path", str(preview_path))
                attachment.setdefault("poster_path", str(poster_path))
        return jsonify({"ok": True, "session_id": session_id, "messages": messages})
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/agent/monitoring/start")
def api_agent_monitoring_start():
    try:
        payload = _monitoring_request_body()
        allowed = {
            "session_id",
            "source_id",
            "start_time",
            "end_time",
            "run_duration_seconds",
            "capture_duration_seconds",
            "interval_seconds",
            "zone_id",
            "parameters",
            "max_consecutive_failures",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"请求包含不支持的字段：{', '.join(unknown)}")
        session_id = _monitoring_session_id(payload.pop("session_id", "default"))
        result = get_agent_service().run_skill(
            "start-monitoring-task",
            session_id=session_id,
            arguments=payload,
        )
        body = {
            **result,
            "session_id": session_id,
            "polling": {
                "status_url": "/api/agent/monitoring/status",
                "events_url": "/api/agent/monitoring/events",
                "recommended_interval_ms": 2000,
            },
        }
        return jsonify(body), _monitoring_http_status(result)
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/agent/realtime-inspection/start")
def api_agent_realtime_inspection_start():
    try:
        payload = _monitoring_request_body()
        session_id = _monitoring_session_id(payload.pop("session_id", "default"))
        result = get_agent_service().run_skill(
            "start-realtime-inspection", session_id=session_id, arguments=payload
        )
        return jsonify({**result, "session_id": session_id,
                        "status_url": "/api/agent/realtime-inspection/status",
                        "events_url": "/api/agent/realtime-inspection/events",
                        "polling": {"recommended_interval_ms": 3000}}), _monitoring_http_status(result)
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/agent/realtime-inspection/stop")
def api_agent_realtime_inspection_stop():
    try:
        payload = _monitoring_request_body()
        session_id = _monitoring_session_id(payload.pop("session_id", "default"))
        result = get_agent_service().run_skill(
            "control-realtime-inspection", session_id=session_id,
            arguments={"action": "stop", **payload},
        )
        return jsonify({**result, "session_id": session_id}), _monitoring_http_status(result)
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.get("/api/agent/realtime-inspection/status")
def api_agent_realtime_inspection_status():
    try:
        session_id = _monitoring_session_id(request.args.get("session_id"))
        result = _query_realtime_detail(
            session_id, task_id=str(request.args.get("task_id") or ""),
            source_id=str(request.args.get("source_id") or ""),
            limit=int(request.args.get("limit", 20)), task_only=True,
        )
        return jsonify({**result, "session_id": session_id}), _monitoring_http_status(result)
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.get("/api/agent/realtime-inspection/events")
def api_agent_realtime_inspection_events():
    try:
        session_id = _monitoring_session_id(request.args.get("session_id"))
        limit = int(request.args.get("limit", 20))
        if not 1 <= limit <= 100:
            raise ValueError("limit 必须在 1 到 100 之间")
        as_bool = lambda name: str(request.args.get(name, "")).strip().lower() in {
            "1", "true", "yes", "on"
        }
        result = _query_realtime_detail(
            session_id, task_id=str(request.args.get("task_id") or ""),
            source_id=str(request.args.get("source_id") or ""), limit=limit,
            event_id=str(request.args.get("event_id") or ""),
            after_event_id=str(request.args.get("after_event_id") or ""),
            latest=as_bool("latest"), active_only=as_bool("active_only"),
            events_only=True, compact=True,
        )
        return jsonify({**result, "session_id": session_id}), _monitoring_http_status(result)
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/agent/archive/start")
def api_agent_archive_start():
    try:
        payload = _monitoring_request_body()
        allowed = {"session_id", "source_id", "segment_seconds", "retention_hours"}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"请求包含不支持的字段：{', '.join(unknown)}")
        session_id = _monitoring_session_id(payload.pop("session_id", "default"))
        result = get_agent_service().run_skill(
            "control-stream-archive",
            session_id=session_id,
            arguments={"action": "start", **payload},
        )
        return jsonify({**result, "session_id": session_id}), _monitoring_http_status(result)
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/agent/archive/stop")
def api_agent_archive_stop():
    try:
        payload = _monitoring_request_body()
        allowed = {"session_id", "source_id"}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"请求包含不支持的字段：{', '.join(unknown)}")
        session_id = _monitoring_session_id(payload.pop("session_id", "default"))
        result = get_agent_service().run_skill(
            "control-stream-archive",
            session_id=session_id,
            arguments={"action": "stop", **payload},
        )
        return jsonify({**result, "session_id": session_id}), _monitoring_http_status(result)
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.get("/api/agent/archive/status")
def api_agent_archive_status():
    try:
        session_id = _monitoring_session_id(request.args.get("session_id"))
        source_id = request.args.get("source_id", "").strip().lower()
        if not source_id:
            raise ValueError("source_id 不能为空")
        result = get_agent_service().run_skill(
            "control-stream-archive",
            session_id=session_id,
            arguments={"action": "query", "source_id": source_id, "limit": 1},
        )
        return jsonify({**result, "session_id": session_id}), _monitoring_http_status(result)
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.get("/api/agent/archive/segments")
def api_agent_archive_segments():
    try:
        session_id = _monitoring_session_id(request.args.get("session_id"))
        source_id = request.args.get("source_id", "").strip().lower()
        limit = int(request.args.get("limit", "100"))
        if not source_id:
            raise ValueError("source_id 不能为空")
        if not 1 <= limit <= 1000:
            raise ValueError("limit 必须在 1 到 1000 之间")
        result = get_agent_service().run_skill(
            "control-stream-archive",
            session_id=session_id,
            arguments={"action": "query", "source_id": source_id, "limit": limit},
        )
        return jsonify({**result, "session_id": session_id}), _monitoring_http_status(result)
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/agent/monitoring/stop")
def api_agent_monitoring_stop():
    try:
        payload = _monitoring_request_body()
        allowed = {"session_id", "task_id", "source_id"}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"请求包含不支持的字段：{', '.join(unknown)}")
        session_id = _monitoring_session_id(payload.pop("session_id", "default"))
        result = get_agent_service().run_skill(
            "control-monitoring-task",
            session_id=session_id,
            arguments={"action": "stop", **payload},
        )
        return jsonify({**result, "session_id": session_id}), _monitoring_http_status(result)
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.get("/api/agent/monitoring/status")
def api_agent_monitoring_status():
    try:
        session_id = _monitoring_session_id(request.args.get("session_id"))
        task_id = request.args.get("task_id", "").strip().lower()
        source_id = request.args.get("source_id", "").strip().lower()
        limit = int(request.args.get("limit", "20"))
        if not 1 <= limit <= 100:
            raise ValueError("limit 必须在 1 到 100 之间")
        result = _query_monitoring_detail(
            session_id,
            task_id=task_id,
            source_id=source_id,
            limit=limit,
        )
        if not result.get("ok"):
            return jsonify(result), _monitoring_http_status(result)
        snapshot = _monitoring_snapshot(session_id, result)
        return jsonify({"ok": True, "session_id": session_id, **snapshot})
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.get("/api/agent/monitoring/events")
def api_agent_monitoring_events():
    try:
        session_id = _monitoring_session_id(request.args.get("session_id"))
        task_id = request.args.get("task_id", "").strip().lower()
        source_id = request.args.get("source_id", "").strip().lower()
        after_segment_id = request.args.get("after_segment_id", "").strip().lower()
        limit = int(request.args.get("limit", "50"))
        if not 1 <= limit <= 100:
            raise ValueError("limit 必须在 1 到 100 之间")
        result = _query_monitoring_detail(
            session_id,
            task_id=task_id,
            source_id=source_id,
            limit=limit,
        )
        if not result.get("ok"):
            return jsonify(result), _monitoring_http_status(result)
        snapshot = _monitoring_snapshot(session_id, result)
        segments = list(reversed(snapshot.pop("segments", [])))
        if after_segment_id:
            cursor_index = next(
                (
                    index
                    for index, item in enumerate(segments)
                    if str(item.get("segment_id") or "") == after_segment_id
                ),
                -1,
            )
            if cursor_index >= 0:
                segments = segments[cursor_index + 1 :]
        events = [
            {
                "event_type": "stream_segment",
                "event_id": str(segment.get("segment_id") or ""),
                **segment,
            }
            for segment in segments
        ]
        latest_alarm = snapshot.get("latest_alarm") or {}
        if latest_alarm:
            events.append(
                {
                    "event_type": "alarm",
                    "event_id": str(latest_alarm.get("alarm_id") or ""),
                    **latest_alarm,
                }
            )
        next_cursor = next(
            (
                str(item.get("segment_id") or "")
                for item in reversed(segments)
                if item.get("segment_id")
            ),
            after_segment_id,
        )
        return jsonify(
            {
                "ok": True,
                "session_id": session_id,
                "found": bool(snapshot.get("found")),
                "task_id": snapshot.get("task_id", ""),
                "source_id": snapshot.get("source_id", ""),
                "status": snapshot.get("status", ""),
                "events": events,
                "next_cursor": next_cursor,
                "connection": snapshot.get("connection", {}),
                "current_segment": snapshot.get("current_segment", {}),
                "progress": snapshot.get("progress", {}),
                "latest_alarm": latest_alarm,
                "stop_reason": snapshot.get("stop_reason", ""),
                "started_at": snapshot.get("started_at", ""),
                "ends_at": snapshot.get("ends_at", ""),
                "last_processed_at": snapshot.get("last_processed_at", ""),
                "updated_at": snapshot.get("updated_at", ""),
            }
        )
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/run")
def api_run():
    try:
        command_value = request.form.get("command", "go")
        conf = float(request.form.get("conf", 0.25))
        known_conf = float(request.form.get("known_conf", 0.40))
        imgsz = int(request.form.get("imgsz", 800))
        nms_iou = float(request.form.get("nms_iou", 0.40))
        duplicate_iou = float(request.form.get("duplicate_iou", 0.45))
        max_area_ratio = float(request.form.get("max_area_ratio", 0.65))

        with pipeline_lock:
            image_path = save_uploaded_image()

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
                known_conf=known_conf,
                imgsz=imgsz,
                nms_iou=nms_iou,
                duplicate_iou=duplicate_iou,
                max_area_ratio=max_area_ratio,
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
        conf = float(request.form.get("conf", "0.25"))
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
