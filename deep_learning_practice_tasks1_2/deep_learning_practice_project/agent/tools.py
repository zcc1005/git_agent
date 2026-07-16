from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from project_config import OUTPUTS_DIR, PROJECT_ROOT, YOLO_MODEL_PATH
from storage import AlarmRecord, SQLiteHistoryStore


@dataclass(frozen=True)
class VideoDetectionOutcome:
    detection: Dict[str, Any]
    alarm_document: Dict[str, Any]
    alarm_report: str
    result_json: str = ""
    alarm_json: str = ""
    alarm_report_path: str = ""


@dataclass(frozen=True)
class ImageDetectionOutcome:
    detection: Dict[str, Any]
    alarm_document: Dict[str, Any]
    alarm_report: str
    result_json: str = ""
    alarm_json: str = ""
    alarm_report_path: str = ""
    visualization_dir: str = ""
    visualization_image: str = ""


VideoDetectionRunner = Callable[[Path, datetime, Dict[str, Any]], VideoDetectionOutcome]
ImageDetectionRunner = Callable[[Path, Dict[str, Any]], ImageDetectionOutcome]
AlarmControlHandler = Callable[[str, AlarmRecord], None]


RISK_NAMES = {
    "none": "无报警",
    "low": "低风险",
    "medium": "中风险",
    "high": "高风险",
}


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class AgentTools:
    """Business tools exposed to the intent router.

    Detection imports are lazy, so listing history or chatting never loads
    OpenCV, Ultralytics, or model weights.  A runner can be injected for tests or
    for a future asynchronous job queue.
    """

    def __init__(
        self,
        store: SQLiteHistoryStore,
        *,
        detection_runner: Optional[VideoDetectionRunner] = None,
        image_detection_runner: Optional[ImageDetectionRunner] = None,
        alarm_control_handler: Optional[AlarmControlHandler] = None,
        now: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.store = store
        self._detection_runner = detection_runner or self._run_existing_video_pipeline
        self._image_detection_runner = (
            image_detection_runner or self._run_existing_image_pipeline
        )
        self._alarm_control_handler = alarm_control_handler
        self._now = now or (lambda: datetime.now().astimezone())

    def detect_image(self, session_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        raw_path = context.get("image_path")
        if not raw_path:
            return {
                "ok": False,
                "requires_attachment": True,
                "reply": "请先上传或选择要检测的图片，然后再发送“检测这张图片”。",
                "data": {},
            }

        image_path = Path(str(raw_path)).expanduser()
        if not image_path.is_file():
            return {
                "ok": False,
                "requires_attachment": True,
                "reply": f"找不到待检测图片：{image_path}",
                "data": {"image_path": str(image_path)},
            }

        parameters = dict(context.get("parameters") or {})
        outcome = self._image_detection_runner(image_path, parameters)
        detection_record, alarm_record = self.store.record_detection(
            session_id,
            source_type="image",
            source_path=str(image_path),
            detection=outcome.detection,
            alarm_document=outcome.alarm_document,
            alarm_report=outcome.alarm_report,
        )
        detection_count = _integer(outcome.detection.get("num_detections"))
        candidate_count = _integer(outcome.detection.get("num_candidates"))
        risk_name = RISK_NAMES.get(alarm_record.risk_level, alarm_record.risk_level)
        return {
            "ok": True,
            "requires_attachment": False,
            "reply": (
                f"图片检测完成：确认 {detection_count} 个异物，"
                f"保留 {candidate_count} 个待确认候选，总体为{risk_name}。"
                f"报警编号 {alarm_record.id}。"
            ),
            "data": {
                "detection_id": detection_record.id,
                "alarm_id": alarm_record.id,
                "risk_level": alarm_record.risk_level,
                "alarm_status": alarm_record.status,
                "detection_count": detection_count,
                "candidate_count": candidate_count,
                "class_counts": outcome.detection.get("class_counts") or {},
                "candidate_counts": outcome.detection.get("candidate_counts") or {},
                "result_json": outcome.result_json,
                "alarm_json": outcome.alarm_json,
                "alarm_report_path": outcome.alarm_report_path,
                "visualization_dir": outcome.visualization_dir,
                "visualization_image": outcome.visualization_image,
            },
        }

    def detect_video(self, session_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        raw_path = context.get("video_path")
        if not raw_path:
            return {
                "ok": False,
                "requires_attachment": True,
                "reply": "请先上传或选择要检测的视频，然后再发送“检测这段视频”。",
                "data": {},
            }

        video_path = Path(str(raw_path)).expanduser()
        if not video_path.is_file():
            return {
                "ok": False,
                "requires_attachment": True,
                "reply": f"找不到待检测视频：{video_path}",
                "data": {"video_path": str(video_path)},
            }

        start_value = context.get("video_start_time")
        if isinstance(start_value, datetime):
            video_start = start_value
        elif start_value:
            video_start = datetime.fromisoformat(str(start_value))
        else:
            video_start = self._now()

        parameters = dict(context.get("parameters") or {})
        outcome = self._detection_runner(video_path, video_start, parameters)
        detection_record, alarm_record = self.store.record_detection(
            session_id,
            source_type="video",
            source_path=str(video_path),
            detection=outcome.detection,
            alarm_document=outcome.alarm_document,
            alarm_report=outcome.alarm_report,
        )
        event_count = _integer(
            outcome.detection.get("num_events"),
            len(outcome.detection.get("events") or []),
        )
        risk_name = RISK_NAMES.get(alarm_record.risk_level, alarm_record.risk_level)
        return {
            "ok": True,
            "requires_attachment": False,
            "reply": (
                f"视频检测完成：发现 {event_count} 个异物事件，"
                f"总体为{risk_name}。报警编号 {alarm_record.id}。"
            ),
            "data": {
                "detection_id": detection_record.id,
                "alarm_id": alarm_record.id,
                "risk_level": alarm_record.risk_level,
                "alarm_status": alarm_record.status,
                "event_count": event_count,
                "class_counts": outcome.detection.get("class_counts") or {},
                "result_json": outcome.result_json,
                "alarm_json": outcome.alarm_json,
                "alarm_report_path": outcome.alarm_report_path,
            },
        }

    def previous_result(self, session_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        record = self.store.latest_detection(session_id)
        if record is None and context.get("allow_global_fallback", True):
            record = self.store.latest_detection()
        if record is None:
            return {
                "ok": True,
                "reply": "还没有历史检测结果。请先上传一段视频并开始检测。",
                "data": {"found": False},
            }

        event_count = self._event_count(record.source_type, record.summary)
        detection_count = _integer(record.summary.get("num_detections"))
        risk_name = RISK_NAMES.get(record.risk_level, record.risk_level)
        return {
            "ok": True,
            "reply": (
                f"上一轮检测于 {record.created_at} 完成，共发现 {event_count} 个异物事件，"
                f"总体为{risk_name}。"
            ),
            "data": {
                "found": True,
                "detection_id": record.id,
                "source_type": record.source_type,
                "source_path": record.source_path,
                "status": record.status,
                "risk_level": record.risk_level,
                "event_count": event_count,
                "detection_count": detection_count,
                "class_counts": record.summary.get("class_counts") or {},
                "created_at": record.created_at,
            },
        }

    def count_high_risk_today(
        self, session_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        target_date = self._today(context)
        count = self.store.count_risk_level("high", target_date)
        return {
            "ok": True,
            "reply": f"{target_date.isoformat()} 共记录 {count} 次高风险报警。",
            "data": {"date": target_date.isoformat(), "high_risk_count": count},
        }

    def generate_daily_report(
        self, session_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        target_date = self._today(context)
        summary = self.store.daily_summary(target_date)
        risks = summary["risk_counts"]
        statuses = summary["status_counts"]
        report = (
            f"{summary['date']} 风险日报\n"
            f"- 检测轮次：{summary['detection_count']}\n"
            f"- 报警总数：{summary['alarm_count']}\n"
            f"- 高/中/低风险：{risks.get('high', 0)}/"
            f"{risks.get('medium', 0)}/{risks.get('low', 0)}\n"
            f"- 待确认/已确认/已取消：{statuses.get('pending', 0)}/"
            f"{statuses.get('confirmed', 0)}/{statuses.get('cancelled', 0)}"
        )
        return {"ok": True, "reply": report, "data": {**summary, "report": report}}

    def confirm_alarm(self, session_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return self._control_alarm("confirm", session_id, context)

    def cancel_alarm(self, session_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return self._control_alarm("cancel", session_id, context)

    def _control_alarm(
        self, action: str, session_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        alarm_id = str(context.get("alarm_id") or "").strip()
        alarm = self.store.get_alarm(alarm_id) if alarm_id else None
        if alarm is None:
            alarm = self.store.latest_actionable_alarm(session_id)
        if alarm is None:
            return {
                "ok": False,
                "reply": "没有找到可操作的报警，请先完成一次产生报警的检测。",
                "data": {"found": False},
            }
        if alarm.status == "inactive":
            return {
                "ok": False,
                "reply": f"报警 {alarm.id} 为无风险记录，不需要确认或取消。",
                "data": {"found": True, "alarm_id": alarm.id, "alarm_status": alarm.status},
            }

        if self._alarm_control_handler is not None:
            self._alarm_control_handler(action, alarm)
        updated = self.store.set_alarm_action(alarm.id, session_id, action)
        action_text = "确认" if action == "confirm" else "取消"
        return {
            "ok": True,
            "reply": f"已{action_text}报警 {updated.id}。",
            "data": {
                "found": True,
                "alarm_id": updated.id,
                "alarm_status": updated.status,
                "risk_level": updated.risk_level,
            },
        }

    def _today(self, context: Dict[str, Any]) -> date:
        raw_date = context.get("date")
        if isinstance(raw_date, date):
            return raw_date
        if raw_date:
            return date.fromisoformat(str(raw_date))
        return self._now().date()

    @staticmethod
    def _event_count(source_type: str, detection: Dict[str, Any]) -> int:
        if source_type == "image":
            return 1 if bool(detection.get("has_foreign_object")) else 0
        return _integer(detection.get("num_events"), len(detection.get("events") or []))

    @staticmethod
    def _run_existing_image_pipeline(
        image_path: Path,
        parameters: Dict[str, Any],
    ) -> ImageDetectionOutcome:
        from task2_yolo.detect_yolo import detect_yiwu
        from task3_alarm.alarm_rule_engine import complete_detection_alarm

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = OUTPUTS_DIR / "agent_image_detections" / f"{image_path.stem}_{timestamp}"
        result_json = output_dir / "detection_results.json"
        allowed_parameters = {
            "conf",
            "known_conf",
            "imgsz",
            "nms_iou",
            "duplicate_iou",
            "duplicate_containment",
            "cross_class_iou",
            "cross_class_containment",
            "max_area_ratio",
            "confirm_low_confidence_unknown",
        }
        effective_parameters = {
            key: value for key, value in parameters.items() if key in allowed_parameters
        }
        detect_yiwu(
            source=image_path,
            model_path=YOLO_MODEL_PATH,
            output_json=result_json,
            **effective_parameters,
        )
        detection = json.loads(result_json.read_text(encoding="utf-8"))
        alarm_json = output_dir / "unified_alarm.json"
        alarm_report_path = output_dir / "alarm_report.txt"
        alarm_document, alarm_report = complete_detection_alarm(
            detection,
            input_json=result_json,
            output_json=alarm_json,
            output_txt=alarm_report_path,
            source_type="image",
        )
        visualization_dir = output_dir / "detections_vis"
        visualization_image = visualization_dir / image_path.name
        return ImageDetectionOutcome(
            detection=detection,
            alarm_document=alarm_document,
            alarm_report=alarm_report,
            result_json=AgentTools._display_path(result_json),
            alarm_json=AgentTools._display_path(alarm_json),
            alarm_report_path=AgentTools._display_path(alarm_report_path),
            visualization_dir=AgentTools._display_path(visualization_dir),
            visualization_image=AgentTools._display_path(visualization_image),
        )

    @staticmethod
    def _run_existing_video_pipeline(
        video_path: Path,
        video_start: datetime,
        parameters: Dict[str, Any],
    ) -> VideoDetectionOutcome:
        # Heavy CV modules are imported only when the detection tool is called.
        from task3_alarm.alarm_rule_engine import complete_detection_alarm
        from video_detection import detect_video_foreign_objects

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = OUTPUTS_DIR / "agent_video_detections" / f"{video_path.stem}_{timestamp}"
        allowed_parameters = {
            "sample_fps",
            "conf",
            "known_conf",
            "imgsz",
            "nms_iou",
            "agnostic_nms",
            "duplicate_iou",
            "duplicate_containment",
            "event_silence_seconds",
            "track_max_age_seconds",
            "min_unknown_hits",
            "unknown_single_frame_conf",
            "track_iou",
            "track_center_distance_ratio",
            "roi",
        }
        effective_parameters = {
            key: value for key, value in parameters.items() if key in allowed_parameters
        }
        detection = detect_video_foreign_objects(
            video_path=video_path,
            model_path=YOLO_MODEL_PATH,
            output_dir=output_dir,
            video_start=video_start,
            **effective_parameters,
        )
        result_json = output_dir / "detection_results.json"
        alarm_json = output_dir / "unified_alarm.json"
        alarm_report_path = output_dir / "alarm_report.txt"
        alarm_document, alarm_report = complete_detection_alarm(
            detection,
            input_json=result_json,
            output_json=alarm_json,
            output_txt=alarm_report_path,
            source_type="video",
        )
        return VideoDetectionOutcome(
            detection=detection,
            alarm_document=alarm_document,
            alarm_report=alarm_report,
            result_json=AgentTools._display_path(result_json),
            alarm_json=AgentTools._display_path(alarm_json),
            alarm_report_path=AgentTools._display_path(alarm_report_path),
        )

    @staticmethod
    def _display_path(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)
