from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence

from agent.monitoring import ACTIVE_MONITORING_STATUSES, MonitoringTaskManager
from agent.realtime_inspection import (
    ACTIVE_STATUSES as ACTIVE_REALTIME_INSPECTION_STATUSES,
    ActiveEvent,
    RealtimeInspectionError,
    RealtimeInspectionManager,
)
from agent.streaming import (
    RtspStreamCapture,
    RtspStreamProbe,
    StreamCaptureResult,
    StreamProbeResult,
)
from agent.video_sources import (
    LongVideoSource,
    LongVideoSourceRegistry,
    load_video_source_registry,
)
from project_config import OUTPUTS_DIR, PROJECT_ROOT, YOLO_MODEL_PATH
from storage import AlarmRecord, RealtimeInspectionTaskRecord, SQLiteHistoryStore

from .archive import HistoricalStreamArchiveManager


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
VideoSegmenter = Callable[[Path, float, Optional[float]], Path]
StreamProbeRunner = Callable[
    [LongVideoSource, Optional[Mapping[str, str]]],
    StreamProbeResult,
]
StreamCaptureRunner = Callable[
    [LongVideoSource, Optional[float], Optional[Mapping[str, str]]],
    StreamCaptureResult,
]
VideoSourceRegistryLoader = Callable[[], LongVideoSourceRegistry]


class DetectionExplainer(Protocol):
    """Narrow interface used by tools without coupling them to a specific LLM SDK."""

    def summarize_detection(self, facts: Mapping[str, Any]) -> str:
        ...

    def explain_detection(
        self,
        question: str,
        question_type: str,
        facts: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
    ) -> str:
        ...


RISK_NAMES = {
    "none": "无报警",
    "low": "低风险",
    "medium": "中风险",
    "high": "高风险",
}

ALARM_STATUS_NAMES = {
    "pending": "待确认",
    "confirmed": "已确认",
    "cancelled": "已取消",
    "inactive": "未触发",
}
QUICK_DETECTION_QUESTIONS = [
    "为什么是高风险？",
    "有什么处置建议？",
    "查看同类历史",
    "解释目标位置",
]


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_datetime_text(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).strip())
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed.isoformat(timespec="seconds")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
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
        video_segmenter: Optional[VideoSegmenter] = None,
        stream_probe_runner: Optional[StreamProbeRunner] = None,
        stream_capture_runner: Optional[StreamCaptureRunner] = None,
        video_source_registry_loader: Optional[VideoSourceRegistryLoader] = None,
        monitoring_manager: Optional[MonitoringTaskManager] = None,
        archive_manager: Optional[HistoricalStreamArchiveManager] = None,
        realtime_inspection_manager: Optional[RealtimeInspectionManager] = None,
        detection_explainer: Optional[DetectionExplainer] = None,
        now: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.store = store
        self._detection_runner = detection_runner or self._run_existing_video_pipeline
        self._image_detection_runner = (
            image_detection_runner or self._run_existing_image_pipeline
        )
        self._alarm_control_handler = alarm_control_handler
        self._video_segmenter = video_segmenter or self._extract_video_segment
        self._stream_probe_runner = stream_probe_runner or RtspStreamProbe()
        self._stream_capture_runner = stream_capture_runner or RtspStreamCapture()
        self._video_source_registry_loader = (
            video_source_registry_loader or load_video_source_registry
        )
        self._detection_explainer = detection_explainer
        self._realtime_summary_slot = threading.BoundedSemaphore(1)
        self._now = now or (lambda: datetime.now().astimezone())
        # Flask 请求与后台监控线程共享模型实例时，串行化重型推理，
        # 避免同一进程内同时占用 GPU/模型状态。
        self._detection_lock = threading.RLock()
        self._monitoring_manager = monitoring_manager or MonitoringTaskManager(
            self.store,
            self.detect_video_source,
            now=self._now,
        )
        self._archive_manager = archive_manager or HistoricalStreamArchiveManager(
            self.store,
            self.capture_video_source,
            now=self._now,
        )
        self._realtime_inspection_manager = realtime_inspection_manager or RealtimeInspectionManager(
            self.store,
            event_sink=self._persist_realtime_event,
            detection_lock=self._detection_lock,
            now=self._now,
        )

    def current_time(self) -> datetime:
        """Return the clock used by tools so planning and execution share one time source."""
        current = self._now()
        return current if current.tzinfo is not None else current.astimezone()

    def set_detection_explainer(
        self, explainer: Optional[DetectionExplainer]
    ) -> None:
        """Attach or clear the optional LLM explanation adapter."""
        self._detection_explainer = explainer

    @staticmethod
    def _deduplicated_text(values: Sequence[Any]) -> list[str]:
        result: list[str] = []
        for value in values:
            text = str(value or "").strip()
            if text and text not in result:
                result.append(text)
        return result

    def detection_facts(
        self,
        detection_id: str,
        *,
        session_id: str = "",
        source_name: str = "",
        representative_frames: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load the authoritative fact whitelist for one persisted detection."""
        record = self.store.get_detection(detection_id)
        if record is None or (session_id and record.session_id != session_id):
            return None
        alarm = self.store.get_alarm_for_detection(record.id)
        document = dict(alarm.report) if alarm is not None else {}
        overall = document.get("overall_risk") or {}
        generated = document.get("generated_report") or {}
        events = [item for item in document.get("events") or [] if isinstance(item, Mapping)]
        document_summary = document.get("detection_summary") or {}
        class_counts = dict(
            document_summary.get("class_counts")
            or record.summary.get("class_counts")
            or {}
        )
        positions: list[Dict[str, Any]] = []
        confidences: list[float] = []
        risk_reasons: list[Any] = []
        for event in events:
            event_risk = event.get("risk") or {}
            risk_reasons.append(event_risk.get("reason"))
            event_id = _integer(event.get("event_id"), len(positions) + 1)
            for obj in event.get("objects") or []:
                if not isinstance(obj, Mapping):
                    continue
                confidence = _float(obj.get("confidence"), -1.0)
                if confidence >= 0:
                    confidences.append(confidence)
                positions.append(
                    {
                        "event_id": event_id,
                        "class_name": str(obj.get("class_name") or obj.get("class") or "未知异物"),
                        "confidence": confidence if confidence >= 0 else None,
                        "position": str(obj.get("position") or "未知区域"),
                        "bbox_xyxy": list(obj.get("bbox_xyxy") or []),
                    }
                )
            event_summary = event.get("detection_summary") or {}
            if event_summary.get("max_confidence") is not None:
                confidences.append(_float(event_summary.get("max_confidence")))
        risk_reasons.append(overall.get("reason"))

        frames = [dict(item) for item in representative_frames or [] if isinstance(item, Mapping)]
        if not frames:
            frames = (
                self.video_event_frames(record.summary)
                if record.source_type in {"video", "realtime"}
                else self.image_event_frames(
                    record.summary,
                    str(record.summary.get("visualization_image") or ""),
                )
            )
        if not frames:
            for index, event in enumerate(events, start=1):
                key_frame = str(event.get("key_frame") or "")
                if key_frame:
                    frames.append(
                        {
                            "event_id": _integer(event.get("event_id"), index),
                            "key_frame": key_frame,
                        }
                    )

        source = document.get("source") or {}
        resolved_source_name = str(source_name or source.get("name") or "").strip()
        if not resolved_source_name:
            raw_source = str(record.source_path or "")
            resolved_source_name = Path(raw_source).name or raw_source or "未知监控源"
        detected_at = str(
            record.source_started_at
            or source.get("start_real_time")
            or record.created_at
        )
        recommended_actions = self._deduplicated_text(
            [generated.get("recommended_action")]
        )
        event_count = _integer(
            document_summary.get("event_count"),
            self._event_count(record.source_type, record.summary),
        )
        object_count = sum(max(0, _integer(value)) for value in class_counts.values())
        if object_count == 0:
            object_count = _integer(document_summary.get("detection_box_count"))
        risk_level = str(
            (alarm.risk_level if alarm is not None else "")
            or record.risk_level
            or overall.get("level")
            or "none"
        ).lower()
        alarm_status = str(alarm.status if alarm is not None else "inactive")
        return {
            "detection_id": record.id,
            "source_type": record.source_type,
            "monitor_source": resolved_source_name,
            "line_id": record.line_id,
            "detected_at": detected_at,
            "source_ended_at": record.source_ended_at,
            "class_counts": class_counts,
            "object_count": object_count,
            "event_count": event_count,
            "max_confidence": max(confidences) if confidences else None,
            "risk_level": risk_level,
            "risk_level_name": RISK_NAMES.get(risk_level, risk_level),
            "risk_reasons": self._deduplicated_text(risk_reasons),
            "recommended_actions": recommended_actions,
            "alarm_status": alarm_status,
            "alarm_status_name": ALARM_STATUS_NAMES.get(alarm_status, alarm_status),
            "positions": positions,
            "representative_frames": frames,
        }

    @staticmethod
    def _fallback_summary(facts: Mapping[str, Any]) -> str:
        classes = "、".join(
            f"{name}{count}个" for name, count in dict(facts.get("class_counts") or {}).items()
        ) or "未发现确认异物"
        confidence = facts.get("max_confidence")
        confidence_text = f"，最高置信度{float(confidence):.4f}" if confidence is not None else ""
        reason = next(iter(facts.get("risk_reasons") or []), "以确定性规则结果为准")
        action = next(iter(facts.get("recommended_actions") or []), "继续监测并结合代表帧人工复核")
        text = (
            f"本次在{facts.get('monitor_source') or '当前监控源'}检测到{classes}{confidence_text}，"
            f"规则引擎判定为{facts.get('risk_level_name') or facts.get('risk_level')}。"
            f"主要依据是{reason}建议{action}"
        )
        if len(text) < 80:
            text += "正式类别、数量、风险等级和报警状态均以结构化预警结果为准。"
        if len(text) > 150:
            text = text[:149].rstrip("，；。") + "。"
        return text

    @staticmethod
    def _analysis_conflicts_with_facts(
        text: str,
        facts: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]] = (),
    ) -> bool:
        if not text.strip():
            return True
        risk_labels = {"none": "无报警", "low": "低风险", "medium": "中风险", "high": "高风险"}
        expected_risk = str(facts.get("risk_level") or "none")
        for level, label in risk_labels.items():
            if level != expected_risk and label in text:
                return True
        allowed_classes = set(dict(facts.get("class_counts") or {}))
        for item in history:
            allowed_classes.update(dict(item.get("class_counts") or {}))
        known_tokens = {"石块异物", "塑料异物", "金属异物", "木块异物", "未知异物"}
        for token in known_tokens:
            if token in text and not any(token in name for name in allowed_classes):
                return True
        status_labels = {
            "pending": "待确认",
            "confirmed": "已确认",
            "cancelled": "已取消",
            "inactive": "未触发",
        }
        expected_status = str(facts.get("alarm_status") or "inactive")
        for status, label in status_labels.items():
            if status != expected_status and label in text:
                return True
        return False

    def detection_presentation(
        self,
        session_id: str,
        detection_id: str,
        *,
        source_name: str = "",
        representative_frames: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        facts = self.detection_facts(
            detection_id,
            session_id=session_id,
            source_name=source_name,
            representative_frames=representative_frames,
        )
        if facts is None:
            return {}
        analysis, source = self._summarize_detection_facts(facts)
        return {
            "structured_alert": facts,
            "ai_analysis": analysis,
            "analysis_source": source,
            "quick_questions": list(QUICK_DETECTION_QUESTIONS),
        }

    def _summarize_detection_facts(
        self, facts: Mapping[str, Any]
    ) -> tuple[str, str]:
        analysis = ""
        source = "fallback"
        if self._detection_explainer is not None:
            try:
                candidate = str(self._detection_explainer.summarize_detection(facts)).strip()
                if 80 <= len(candidate) <= 150 and not self._analysis_conflicts_with_facts(candidate, facts):
                    analysis = candidate
                    source = "llm"
            except Exception:
                analysis = ""
        if not analysis:
            analysis = self._fallback_summary(facts)
        return analysis, source

    def _similar_detection_history(
        self,
        detection_id: str,
        class_counts: Mapping[str, Any],
        limit: int,
    ) -> list[Dict[str, Any]]:
        class_names = set(str(name) for name in class_counts)
        matches: list[Dict[str, Any]] = []
        for record in self.store.query_detections(limit=max(100, limit * 10)):
            if record.id == detection_id:
                continue
            historical_counts = dict(record.summary.get("class_counts") or {})
            if class_names and not class_names.intersection(historical_counts):
                continue
            alarm = self.store.get_alarm_for_detection(record.id)
            matches.append(
                {
                    "detection_id": record.id,
                    "detected_at": record.source_started_at or record.created_at,
                    "line_id": record.line_id,
                    "risk_level": record.risk_level,
                    "class_counts": historical_counts,
                    "alarm_status": alarm.status if alarm is not None else "inactive",
                }
            )
            if len(matches) >= limit:
                break
        return matches

    @staticmethod
    def _fallback_explanation(
        question_type: str,
        facts: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
    ) -> str:
        if question_type == "risk_reason":
            reasons = "；".join(facts.get("risk_reasons") or []) or "未记录额外风险原因。"
            return f"该记录的权威风险等级为{facts.get('risk_level_name')}。规则依据：{reasons}"
        if question_type == "action_advice":
            actions = "；".join(facts.get("recommended_actions") or []) or "继续监测并结合代表帧人工复核。"
            return f"按规则引擎建议：{actions}报警当前状态为{facts.get('alarm_status_name')}。"
        if question_type == "similar_history":
            if not history:
                return "历史库中暂未查询到与本次异物类别相同的其他检测记录。"
            detail = "；".join(
                f"{item['detected_at']}，{RISK_NAMES.get(str(item['risk_level']), item['risk_level'])}"
                for item in history[:5]
            )
            return f"共找到{len(history)}条同类历史记录。最近记录：{detail}。"
        if question_type == "target_position":
            positions = facts.get("positions") or []
            if not positions:
                return "当前记录没有可用于解释位置的目标框信息，请查看代表帧进行人工确认。"
            detail = "；".join(
                f"事件{item['event_id']}的{item['class_name']}位于{item['position']}，框坐标{item['bbox_xyxy']}"
                for item in positions[:5]
            )
            return f"位置来自检测框与图像尺寸的确定性换算：{detail}。"
        return AgentTools._fallback_summary(facts)

    def explain_detection_result(
        self, session_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        detection_id = str(context.get("detection_id") or "").strip()
        if not detection_id:
            latest = self.store.latest_detection(session_id)
            detection_id = latest.id if latest is not None else ""
        if not detection_id:
            return {
                "ok": False,
                "reply": "当前会话还没有检测记录，请先执行一次图片、视频或监控检测。",
                "data": {"found": False, "needs_detection": True},
            }
        facts = self.detection_facts(detection_id, session_id=session_id)
        if facts is None:
            return {
                "ok": False,
                "reply": "当前会话找不到这条检测记录，请先执行一次检测。",
                "data": {"found": False, "needs_detection": True},
            }
        question_type = str(context.get("question_type") or "general").strip().lower()
        question = str(context.get("question") or "请解释这次检测结果").strip()
        history = (
            self._similar_detection_history(
                detection_id,
                facts.get("class_counts") or {},
                max(1, min(_integer(context.get("history_limit"), 10), 50)),
            )
            if question_type == "similar_history"
            else []
        )
        analysis = ""
        source = "fallback"
        if self._detection_explainer is not None:
            try:
                candidate = str(
                    self._detection_explainer.explain_detection(
                        question, question_type, facts, history
                    )
                ).strip()
                if candidate and not self._analysis_conflicts_with_facts(candidate, facts, history):
                    analysis = candidate
                    source = "llm"
            except Exception:
                analysis = ""
        if not analysis:
            analysis = self._fallback_explanation(question_type, facts, history)
        return {
            "ok": True,
            "reply": analysis,
            "data": {
                "found": True,
                "detection_id": detection_id,
                "question_type": question_type,
                "authoritative_facts": facts,
                "history_summary": history,
                "ai_analysis": analysis,
                "analysis_source": source,
                "quick_questions": list(QUICK_DETECTION_QUESTIONS),
            },
        }

    def video_source_catalog(self) -> list[Dict[str, Any]]:
        """Return planner-safe source aliases without URLs or environment names."""
        try:
            registry = self._video_source_registry_loader()
        except (FileNotFoundError, ValueError):
            return []
        return [
            {
                "source_id": source.source_id,
                "display_name": source.display_name,
                "line_id": source.line_id,
                "source_kind": source.source_kind,
                "default_capture_seconds": (
                    source.stream.capture_window_seconds if source.stream else None
                ),
                "zones": [
                    {
                        "zone_id": zone.zone_id,
                        "display_name": zone.display_name,
                    }
                    for zone in source.zones
                ],
            }
            for source in registry.sources
        ]

    def probe_video_source(
        self,
        session_id: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        del session_id
        source_id = str(context.get("source_id") or "").strip().lower()
        try:
            registry = self._video_source_registry_loader()
        except FileNotFoundError:
            return {
                "ok": False,
                "error_code": "configuration_error",
                "reply": "视频源注册表尚未配置。",
                "data": {"source_id": source_id, "online": False},
            }
        except ValueError:
            return {
                "ok": False,
                "error_code": "configuration_error",
                "reply": "视频源注册表配置无效。",
                "data": {"source_id": source_id, "online": False},
            }
        try:
            source = registry.get(source_id)
        except LookupError:
            return {
                "ok": False,
                "error_code": "source_not_found",
                "reply": f"未找到已注册的视频源：{source_id}",
                "data": {"source_id": source_id, "online": False},
            }

        if not source.is_rtsp:
            return {
                "ok": False,
                "error_code": "not_rtsp_source",
                "reply": f"{source.display_name}不是 RTSP 视频源，无法执行连接探测。",
                "data": {
                    "source_id": source.source_id,
                    "display_name": source.display_name,
                    "line_id": source.line_id,
                    "online": False,
                    "error_code": "not_rtsp_source",
                },
            }

        try:
            result = self._stream_probe_runner(source, None)
            data = result.to_dict()
        except Exception:
            return {
                "ok": False,
                "error_code": "probe_failed",
                "reply": f"{source.display_name}连接探测失败。",
                "data": {
                    "source_id": source.source_id,
                    "display_name": source.display_name,
                    "line_id": source.line_id,
                    "online": False,
                    "error_code": "probe_failed",
                },
            }

        if result.error_code == "configuration_error":
            return {
                "ok": False,
                "error_code": result.error_code,
                "reply": result.error_message,
                "data": data,
            }
        if result.online:
            resolution = (
                f"{result.width}×{result.height}"
                if result.width and result.height
                else "分辨率未知"
            )
            fps = f"{result.fps:g} FPS" if result.fps else "FPS 未知"
            codec = result.codec.upper() if result.codec else "编码未知"
            return {
                "ok": True,
                "reply": (
                    f"{result.display_name}当前在线：{resolution}，{fps}，"
                    f"{codec}，连接延迟 {result.latency_ms} 毫秒。"
                ),
                "data": data,
            }
        return {
            "ok": True,
            "reply": f"{result.display_name}当前离线：{result.error_message}",
            "data": data,
        }

    def capture_video_source(
        self,
        session_id: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        del session_id
        source_id = str(context.get("source_id") or "").strip().lower()
        try:
            registry = self._video_source_registry_loader()
        except (FileNotFoundError, ValueError):
            return {
                "ok": False,
                "error_code": "configuration_error",
                "reply": "视频源注册表尚未正确配置。",
                "data": {"source_id": source_id, "captured": False},
            }
        try:
            source = registry.get(source_id)
        except LookupError:
            return {
                "ok": False,
                "error_code": "source_not_found",
                "reply": f"未找到已注册的视频源：{source_id}",
                "data": {"source_id": source_id, "captured": False},
            }
        if not source.is_rtsp:
            return {
                "ok": False,
                "error_code": "not_rtsp_source",
                "reply": f"{source.display_name}不是 RTSP 视频源，无法执行实时采集。",
                "data": {
                    "source_id": source.source_id,
                    "display_name": source.display_name,
                    "line_id": source.line_id,
                    "captured": False,
                    "error_code": "not_rtsp_source",
                },
            }

        duration = context.get("duration_seconds")
        try:
            result = self._stream_capture_runner(
                source,
                float(duration) if duration is not None else None,
                None,
            )
            data = result.to_dict()
        except Exception:
            return {
                "ok": False,
                "error_code": "capture_failed",
                "reply": f"{source.display_name}视频采集失败。",
                "data": {
                    "source_id": source.source_id,
                    "display_name": source.display_name,
                    "line_id": source.line_id,
                    "captured": False,
                    "error_code": "capture_failed",
                },
            }

        for path_key in ("video_path", "metadata_path"):
            if data.get(path_key):
                data[path_key] = self._display_path(Path(str(data[path_key])))
        if not result.captured:
            return {
                "ok": False,
                "error_code": result.error_code,
                "reply": f"{result.display_name}视频采集失败：{result.error_message}",
                "data": data,
            }
        return {
            "ok": True,
            "reply": (
                f"已从{result.display_name}采集 {result.duration_seconds:g} 秒视频，"
                f"共 {result.frame_count} 帧。"
            ),
            "data": data,
        }

    def detect_video_source(
        self,
        session_id: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        source_id = str(context.get("source_id") or "").strip().lower()
        try:
            registry = self._video_source_registry_loader()
        except (FileNotFoundError, ValueError):
            return {
                "ok": False,
                "error_code": "configuration_error",
                "reply": "视频源注册表尚未正确配置。",
                "data": {"source_id": source_id, "workflow": []},
            }
        try:
            source = registry.get(source_id)
        except LookupError:
            return {
                "ok": False,
                "error_code": "source_not_found",
                "reply": f"未找到已注册的视频源：{source_id}",
                "data": {"source_id": source_id, "workflow": []},
            }
        if not source.is_rtsp:
            return {
                "ok": False,
                "error_code": "not_rtsp_source",
                "reply": f"{source.display_name}不是 RTSP 视频源，无法执行实时检测。",
                "data": {
                    "source_id": source.source_id,
                    "display_name": source.display_name,
                    "line_id": source.line_id,
                    "workflow": [],
                },
            }

        parameters = dict(context.get("parameters") or {})
        zone_id = str(context.get("zone_id") or "").strip().lower()
        selected_zone: Dict[str, Any] = {}
        if zone_id:
            zone = next((item for item in source.zones if item.zone_id == zone_id), None)
            if zone is None:
                return {
                    "ok": False,
                    "error_code": "zone_not_found",
                    "reply": f"视频源 {source.display_name} 未注册区域：{zone_id}",
                    "data": {
                        "source_id": source.source_id,
                        "display_name": source.display_name,
                        "line_id": source.line_id,
                        "zone_id": zone_id,
                        "available_zones": [item.zone_id for item in source.zones],
                        "workflow": [],
                    },
                }
            parameters["roi"] = zone.roi
            selected_zone = {
                "zone_id": zone.zone_id,
                "display_name": zone.display_name,
                "roi": list(zone.roi),
            }

        capture_arguments: Dict[str, Any] = {"source_id": source.source_id}
        if context.get("duration_seconds") is not None:
            capture_arguments["duration_seconds"] = context["duration_seconds"]
        capture_result = self.capture_video_source(session_id, capture_arguments)
        capture_data = dict(capture_result.get("data") or {})
        if not capture_result.get("ok"):
            return {
                "ok": False,
                "error_code": str(capture_result.get("error_code") or "capture_failed"),
                "reply": capture_result.get("reply") or "RTSP 视频采集失败。",
                "data": {
                    "source_id": source.source_id,
                    "display_name": source.display_name,
                    "line_id": source.line_id,
                    "zone": selected_zone,
                    "capture": capture_data,
                    "workflow": ["capture-video-source"],
                },
            }

        captured_path = Path(str(capture_data.get("video_path") or ""))
        if not captured_path.is_absolute():
            captured_path = PROJECT_ROOT / captured_path
        detection_context = {
            "video_path": str(captured_path),
            "video_start_time": capture_data.get("started_at"),
            "source_ended_at": capture_data.get("ended_at"),
            "line_id": source.line_id,
            "parameters": parameters,
        }
        try:
            detection_result = self.detect_video(session_id, detection_context)
        except Exception:
            return {
                "ok": False,
                "error_code": "detection_failed",
                "reply": f"{source.display_name}视频已采集，但异物检测执行失败。",
                "data": {
                    "source_id": source.source_id,
                    "display_name": source.display_name,
                    "line_id": source.line_id,
                    "zone": selected_zone,
                    "capture": capture_data,
                    "workflow": ["capture-video-source", "detect-video"],
                },
            }

        data = dict(detection_result.get("data") or {})
        data.update(
            {
                "source_id": source.source_id,
                "display_name": source.display_name,
                "line_id": source.line_id,
                "zone": selected_zone,
                "capture": capture_data,
                "workflow": [
                    "capture-video-source",
                    "detect-video",
                    "assess-risk",
                    "persist-history",
                    "create-alarm",
                ],
            }
        )
        if not detection_result.get("ok"):
            return {
                "ok": False,
                "error_code": str(
                    detection_result.get("error_code") or "detection_failed"
                ),
                "reply": detection_result.get("reply") or "异物检测执行失败。",
                "data": data,
                "requires_attachment": bool(
                    detection_result.get("requires_attachment")
                ),
            }
        detection_reply = str(detection_result.get("reply") or "任务已完成。")
        if detection_reply.startswith("视频检测完成："):
            detection_reply = detection_reply.removeprefix("视频检测完成：")
        return {
            "ok": True,
            "reply": f"{source.display_name}实时检测完成：{detection_reply}",
            "data": data,
        }

    def start_monitoring_task(
        self,
        session_id: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        source_id = str(context.get("source_id") or "").strip().lower()
        try:
            registry = self._video_source_registry_loader()
            source = registry.get(source_id)
        except FileNotFoundError:
            return {
                "ok": False,
                "error_code": "configuration_error",
                "reply": "视频源注册表尚未配置。",
                "data": {},
            }
        except ValueError:
            return {
                "ok": False,
                "error_code": "configuration_error",
                "reply": "视频源注册表配置无效。",
                "data": {},
            }
        except LookupError:
            return {
                "ok": False,
                "error_code": "source_not_found",
                "reply": f"未找到已注册的视频源：{source_id}",
                "data": {},
            }
        if not source.is_rtsp or source.stream is None:
            return {
                "ok": False,
                "error_code": "not_rtsp_source",
                "reply": f"{source.display_name}不是 RTSP 视频源，无法启动监控任务。",
                "data": {},
            }

        zone_id = str(context.get("zone_id") or "").strip().lower()
        if zone_id and not any(zone.zone_id == zone_id for zone in source.zones):
            return {
                "ok": False,
                "error_code": "zone_not_found",
                "reply": f"视频源 {source.display_name} 未注册区域：{zone_id}",
                "data": {
                    "available_zones": [zone.zone_id for zone in source.zones]
                },
            }

        current = self.current_time()
        start_value = context.get("start_time")
        start = (
            datetime.fromisoformat(str(start_value).replace("Z", "+00:00"))
            if start_value
            else current
        )
        if start.tzinfo is None:
            return {
                "ok": False,
                "error_code": "invalid_schedule",
                "reply": "start_time 必须包含时区。",
                "data": {},
            }
        end_value = context.get("end_time")
        run_duration = context.get("run_duration_seconds")
        if end_value:
            end = datetime.fromisoformat(str(end_value).replace("Z", "+00:00"))
        else:
            end = start + timedelta(seconds=float(run_duration))
        if end.tzinfo is None:
            return {
                "ok": False,
                "error_code": "invalid_schedule",
                "reply": "end_time 必须包含时区。",
                "data": {},
            }
        config = {
            "capture_duration_seconds": float(
                context.get("capture_duration_seconds")
                or source.stream.capture_window_seconds
            ),
            "interval_seconds": float(context.get("interval_seconds", 60.0)),
            "max_consecutive_failures": int(
                context.get("max_consecutive_failures", 3)
            ),
            "parameters": dict(context.get("parameters") or {}),
        }
        try:
            task = self._monitoring_manager.start_task(
                session_id,
                source_id=source.source_id,
                line_id=source.line_id,
                zone_id=zone_id,
                start_time=start,
                end_time=end,
                config=config,
            )
        except ValueError as exc:
            return {
                "ok": False,
                "error_code": "invalid_schedule",
                "reply": str(exc),
                "data": {},
            }
        job = self.store.get_monitoring_job(task.id)
        return {
            "ok": True,
            "reply": (
                f"已创建{source.display_name}非全天候监控任务，"
                f"计划从 {task.start_time} 运行到 {task.end_time}。"
            ),
            "data": {
                **task.to_dict(),
                "monitoring_job": job.to_dict() if job else {},
            },
        }

    def control_monitoring_task(
        self,
        session_id: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        action = str(context.get("action") or "query").strip().lower()
        task_id = str(context.get("task_id") or "").strip().lower()
        source_id = str(context.get("source_id") or "").strip().lower()
        limit = int(context.get("limit", 10))
        if action == "query":
            if task_id:
                task = self.store.get_monitoring_task(task_id)
                if task is None or task.session_id != session_id:
                    return {
                        "ok": False,
                        "error_code": "task_not_found",
                        "reply": f"找不到当前会话的监控任务：{task_id}",
                        "data": {},
                    }
                runs = self.store.list_monitoring_runs(task.id, limit=limit)
                job = self.store.get_monitoring_job(task.id)
                segments = self.store.list_stream_segments(task.id, limit=limit)
                return {
                    "ok": True,
                    "reply": (
                        f"监控任务 {task.id} 当前状态为 "
                        f"{job.status if job else task.status}，"
                        f"已完成 {task.runs_completed} 轮检测。"
                    ),
                    "data": {
                        "found": True,
                        "task": task.to_dict(),
                        "monitoring_job": job.to_dict() if job else {},
                        "runs": [run.to_dict() for run in runs],
                        "segments": [segment.to_dict() for segment in segments],
                    },
                }
            tasks = self.store.list_monitoring_tasks(
                session_id=session_id,
                source_id=source_id,
                limit=limit,
            )
            return {
                "ok": True,
                "reply": (
                    f"当前会话共有 {len(tasks)} 条监控任务记录。"
                    if tasks
                    else "当前会话还没有监控任务。"
                ),
                "data": {
                    "found": bool(tasks),
                    "tasks": [task.to_dict() for task in tasks],
                    "monitoring_jobs": [
                        job.to_dict()
                        for task in tasks
                        for job in [self.store.get_monitoring_job(task.id)]
                        if job is not None
                    ],
                },
            }

        if not task_id:
            active = self.store.list_monitoring_tasks(
                session_id=session_id,
                source_id=source_id,
                statuses=ACTIVE_MONITORING_STATUSES,
                limit=1,
            )
            if not active:
                return {
                    "ok": False,
                    "error_code": "task_not_found",
                    "reply": "当前会话没有可停止的监控任务。",
                    "data": {},
                }
            task_id = active[0].id
        try:
            task = self._monitoring_manager.stop_task(
                task_id,
                session_id=session_id,
            )
        except LookupError as exc:
            return {
                "ok": False,
                "error_code": "task_not_found",
                "reply": str(exc),
                "data": {},
            }
        job = self.store.get_monitoring_job(task.id)
        return {
            "ok": True,
            "reply": f"已请求停止监控任务 {task.id}；当前轮完成后不会启动下一轮。",
            "data": {
                **task.to_dict(),
                "monitoring_job": job.to_dict() if job else {},
            },
        }

    def start_realtime_inspection(self, session_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        forbidden = {"rtsp_url", "line_id", "output_path", "model_path", "command", "python_code"}
        supplied_forbidden = sorted(forbidden.intersection(context))
        if supplied_forbidden:
            return {"ok": False, "error_code": "invalid_parameters",
                    "reply": f"实时巡检不允许传入字段：{', '.join(supplied_forbidden)}。", "data": {}}
        source_id = str(context.get("source_id") or "").strip().lower()
        try:
            source = self._video_source_registry_loader().get(source_id)
        except (FileNotFoundError, ValueError):
            return {"ok": False, "error_code": "configuration_error", "reply": "视频源注册表配置无效。", "data": {}}
        except LookupError:
            return {"ok": False, "error_code": "source_not_found", "reply": f"未找到已注册的视频源：{source_id}", "data": {}}
        if not source.is_rtsp or source.stream is None:
            return {"ok": False, "error_code": "not_rtsp_source", "reply": f"{source.display_name}不是 RTSP 视频源。", "data": {}}
        end_value = context.get("end_time")
        duration_value = context.get("run_duration_seconds")
        if bool(end_value) == bool(duration_value):
            return {"ok": False, "error_code": "invalid_schedule",
                    "reply": "必须且只能提供 end_time 或 run_duration_seconds 作为结束条件。", "data": {},
                    "needs_clarification": not end_value and not duration_value}
        current = self.current_time()
        try:
            start = datetime.fromisoformat(str(context.get("start_time")).replace("Z", "+00:00")) if context.get("start_time") else current
            if start.tzinfo is None: raise ValueError("start_time 必须包含时区。")
            if start < current - timedelta(seconds=1): raise ValueError("历史时间不能创建实时巡检任务。")
            end = (datetime.fromisoformat(str(end_value).replace("Z", "+00:00"))
                   if end_value else start + timedelta(seconds=float(duration_value)))
            if end.tzinfo is None: raise ValueError("end_time 必须包含时区。")
        except (TypeError, ValueError) as exc:
            return {"ok": False, "error_code": "invalid_schedule", "reply": str(exc), "data": {}}
        parameters = dict(context.get("parameters") or {})
        forbidden_parameters = forbidden.intersection(parameters)
        if forbidden_parameters:
            return {"ok": False, "error_code": "invalid_parameters",
                    "reply": f"parameters 不允许包含：{', '.join(sorted(forbidden_parameters))}。", "data": {}}
        zone_id = str(context.get("zone_id") or "").strip().lower()
        if zone_id and parameters.get("roi") is not None:
            return {"ok": False, "error_code": "invalid_parameters", "reply": "zone_id 与 parameters.roi 不能同时提供。", "data": {}}
        selected_zone: Dict[str, Any] = {}
        if zone_id:
            zone = next((item for item in source.zones if item.zone_id == zone_id), None)
            if zone is None:
                return {"ok": False, "error_code": "zone_not_found", "reply": f"视频源未注册区域：{zone_id}",
                        "data": {"available_zones": [item.zone_id for item in source.zones]}}
            parameters["roi"] = list(zone.roi)
            selected_zone = {"zone_id": zone.zone_id, "display_name": zone.display_name, "roi": list(zone.roi)}
        config = {
            "parameters": parameters,
            "reconnect_interval_seconds": float(context.get("reconnect_interval_seconds", 3.0)),
            "max_consecutive_failures": int(context.get("max_consecutive_failures", 3)),
            "min_event_hits": int(context.get("min_event_hits", 2)),
            "event_silence_seconds": float(context.get("event_silence_seconds", 1.0)),
        }
        try:
            task = self._realtime_inspection_manager.start_task(
                session_id, source=source, start_time=start, end_time=end, zone_id=zone_id,
                sample_fps=float(context.get("sample_fps", 2.0)), config=config,
            )
        except RealtimeInspectionError as exc:
            return {"ok": False, "error_code": exc.code, "reply": str(exc), "data": {}}
        return {"ok": True, "reply": f"正在启动{source.display_name}实时巡检，任务 {task.id}。",
                "data": {**task.to_dict(), "display_name": source.display_name, "zone": selected_zone}}

    def control_realtime_inspection(self, session_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        action = str(context.get("action") or "query").strip().lower()
        action = {"view": "query", "show": "query", "status": "query", "get": "query", "cancel": "stop"}.get(action, action)
        task_id = str(context.get("task_id") or "").strip().lower()
        source_id = str(context.get("source_id") or "").strip().lower()
        limit = max(1, min(int(context.get("limit", 10)), 100))
        event_id = str(context.get("event_id") or "").strip().lower()
        after_event_id = str(context.get("after_event_id") or "").strip().lower()
        active_only = bool(context.get("active_only", False))
        latest = bool(context.get("latest", False))
        event_query = bool(event_id or after_event_id or active_only or latest or context.get("events_only"))
        if action == "query":
            if event_id:
                selected_event = self.store.get_realtime_inspection_event(event_id)
                task = (
                    self.store.get_realtime_inspection_task(selected_event.task_id)
                    if selected_event is not None else None
                )
                if task is None or task.session_id != session_id:
                    return {"ok": False, "error_code": "event_not_found",
                            "reply": "找不到当前会话的实时巡检事件。", "data": {}}
                task_data = self._realtime_task_display_data(task)
                event_data = self._realtime_event_display_data(selected_event)
                return {"ok": True, "reply": f"已找到事件 {event_id} 的详细报告。",
                        "data": {"found": True, "task": task_data, "events": [event_data],
                                 "next_event_id": event_id}}
            if task_id:
                task = self.store.get_realtime_inspection_task(task_id)
                if task is None or task.session_id != session_id:
                    return {"ok": False, "error_code": "task_not_found", "reply": "找不到当前会话的实时巡检任务。", "data": {}}
                events = self.store.list_realtime_inspection_events(
                    task.id, limit, active_only=active_only,
                    after_event_id=after_event_id, latest=latest,
                )
                if event_query and not after_event_id and not latest:
                    events = list(reversed(events))
                task_data = self._realtime_task_display_data(task)
                report = self._realtime_inspection_report(task_data, events)
                reply = (
                    "当前实时巡检尚未确认异物事件。"
                    if event_query and not events
                    else self._realtime_task_status_reply(task_data)
                )
                event_values = [self._realtime_event_display_data(item) for item in events]
                return {"ok": True, "reply": reply,
                        "data": {"found": True, "task": task_data,
                                 "events": event_values,
                                 "next_event_id": event_values[-1]["event_id"] if event_values else after_event_id,
                                 "realtime_report": report}}
            tasks = self.store.list_realtime_inspection_tasks(session_id=session_id, source_id=source_id, limit=limit)
            if not tasks:
                return {"ok": True, "reply": "当前会话还没有实时巡检任务。",
                        "data": {"found": False, "tasks": []}}
            active = next(
                (item for item in tasks if item.status in ACTIVE_REALTIME_INSPECTION_STATUSES),
                tasks[0],
            )
            events = self.store.list_realtime_inspection_events(
                active.id, limit, active_only=active_only,
                after_event_id=after_event_id, latest=latest,
            )
            if event_query and not after_event_id and not latest:
                events = list(reversed(events))
            task_data = self._realtime_task_display_data(active)
            report = self._realtime_inspection_report(task_data, events)
            event_values = [self._realtime_event_display_data(item) for item in events]
            return {
                "ok": True,
                "reply": (
                    "当前实时巡检尚未确认异物事件。"
                    if event_query and not events
                    else self._realtime_task_status_reply(task_data)
                ),
                "data": {
                    "found": True,
                    "task": task_data,
                    "tasks": [self._realtime_task_display_data(item) for item in tasks],
                    "events": event_values,
                    "next_event_id": event_values[-1]["event_id"] if event_values else after_event_id,
                    "realtime_report": report,
                },
            }
        if action != "stop":
            return {"ok": False, "error_code": "invalid_action", "reply": "action 只能是 query 或 stop。", "data": {}}
        if not task_id:
            active = self.store.list_realtime_inspection_tasks(
                session_id=session_id, source_id=source_id,
                statuses=ACTIVE_REALTIME_INSPECTION_STATUSES, limit=1,
            )
            if not active:
                recent = self.store.list_realtime_inspection_tasks(
                    session_id=session_id, source_id=source_id, limit=1,
                )
                if recent:
                    task_data = self._realtime_task_display_data(recent[0])
                    return {
                        "ok": True,
                        "reply": self._realtime_terminal_stop_reply(task_data),
                        "data": {"found": True, "task": task_data},
                    }
                return {"ok": True, "reply": "当前会话没有正在运行的实时巡检任务，无需停止。",
                        "data": {"found": False}}
            task_id = active[0].id
        existing = self.store.get_realtime_inspection_task(task_id)
        if existing is None or existing.session_id != session_id:
            return {"ok": False, "error_code": "task_not_found", "reply": "找不到当前会话的实时巡检任务。", "data": {}}
        if existing.status not in ACTIVE_REALTIME_INSPECTION_STATUSES:
            task_data = self._realtime_task_display_data(existing)
            return {"ok": True, "reply": self._realtime_terminal_stop_reply(task_data),
                    "data": {"found": True, "task": task_data}}
        try:
            task = self._realtime_inspection_manager.stop_task(task_id, session_id=session_id)
        except RealtimeInspectionError as exc:
            return {"ok": False, "error_code": exc.code, "reply": str(exc), "data": {}}
        return {"ok": True, "reply": f"已请求停止实时巡检任务 {task.id}，当前单帧推理结束后释放资源。", "data": task.to_dict()}

    @staticmethod
    def _realtime_terminal_stop_reply(task: Mapping[str, Any]) -> str:
        status_names = {
            "completed": "已经按计划完成",
            "stopped": "已经停止",
            "failed": "已经执行失败并结束",
            "interrupted": "已经因服务重启而中断",
        }
        source_name = str(task.get("display_name") or task.get("source_id") or "当前监控")
        status = status_names.get(str(task.get("status") or ""), "当前不在运行")
        return f"{source_name}最近一次实时巡检{status}，无需重复停止。"

    def _realtime_task_display_data(
        self, task: RealtimeInspectionTaskRecord
    ) -> Dict[str, Any]:
        data = task.to_dict()
        source = next(
            (
                item
                for item in self.video_source_catalog()
                if str(item.get("source_id") or "") == task.source_id
            ),
            None,
        )
        if source:
            data["display_name"] = str(source.get("display_name") or task.source_id)
        return data

    def _realtime_event_display_data(self, record: Any) -> Dict[str, Any]:
        event = record.to_dict() if hasattr(record, "to_dict") else dict(record)
        alarm = self.store.get_alarm(str(event.get("alarm_id") or ""))
        alarm_status = str(alarm.status if alarm is not None else "inactive")
        metadata = event.get("metadata") if isinstance(event.get("metadata"), Mapping) else {}
        event["alarm_status"] = alarm_status
        event["alarm_status_name"] = ALARM_STATUS_NAMES.get(alarm_status, alarm_status)
        event["risk_level_name"] = RISK_NAMES.get(
            str(event.get("risk_level") or "none"), str(event.get("risk_level") or "none")
        )
        event["representative_frame"] = str(
            event.get("representative_frame") or event.get("image_path") or ""
        )
        event["alarm_report"] = {
            "text": str(event.get("alarm_report") or (alarm.report_text if alarm else "")),
            "document": dict(alarm.report) if alarm is not None else {},
            "json_path": str(metadata.get("alarm_json_path") or ""),
            "report_path": str(metadata.get("alarm_report_path") or ""),
        }
        return event

    def _schedule_realtime_event_summary(
        self, event_id: str, facts: Mapping[str, Any]
    ) -> None:
        if self._detection_explainer is None:
            return
        if not self._realtime_summary_slot.acquire(blocking=False):
            return

        def enrich() -> None:
            try:
                summary, source = self._summarize_detection_facts(facts)
                if source != "llm":
                    return
                current = self.store.get_realtime_inspection_event(event_id)
                if current is None:
                    return
                metadata = dict(current.metadata)
                metadata["analysis_source"] = "llm"
                self.store.update_realtime_inspection_event(
                    event_id, llm_summary=summary, metadata=metadata
                )
            except Exception:
                # The formal event/report is already durable. LLM enrichment is optional.
                return
            finally:
                self._realtime_summary_slot.release()

        threading.Thread(
            target=enrich,
            name=f"realtime-summary-{event_id[-12:]}",
            daemon=True,
        ).start()

    def _realtime_inspection_report(
        self,
        task: Mapping[str, Any],
        events: Sequence[Any],
    ) -> Dict[str, Any]:
        class_counts: Dict[str, int] = {}
        alarm_status_counts: Dict[str, int] = {}
        risk_reasons: list[str] = []
        recommended_actions: list[str] = []
        report_events: list[Dict[str, Any]] = []
        max_confidence: Optional[float] = None

        for index, record in enumerate(events, start=1):
            event = record.to_dict() if hasattr(record, "to_dict") else dict(record)
            class_name = str(event.get("class_name") or "未知异物")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidence = _float(event.get("confidence"), -1.0)
            if confidence >= 0 and (max_confidence is None or confidence > max_confidence):
                max_confidence = confidence

            alarm = self.store.get_alarm(str(event.get("alarm_id") or ""))
            alarm_status = str(alarm.status if alarm is not None else "inactive")
            alarm_status_counts[alarm_status] = alarm_status_counts.get(alarm_status, 0) + 1
            document = dict(alarm.report) if alarm is not None else {}
            generated = document.get("generated_report") or {}
            recommended_actions.extend(
                self._deduplicated_text([generated.get("recommended_action")])
            )
            document_events = [
                item for item in document.get("events") or [] if isinstance(item, Mapping)
            ]
            event_document = document_events[0] if document_events else {}
            event_risk = event_document.get("risk") or {}
            risk_reasons.extend(
                self._deduplicated_text(
                    [event_risk.get("reason"), (document.get("overall_risk") or {}).get("reason")]
                )
            )
            objects = [item for item in event_document.get("objects") or [] if isinstance(item, Mapping)]
            first_object = objects[0] if objects else {}
            report_events.append(
                {
                    "event_number": index,
                    "event_id": str(event.get("event_id") or index),
                    "detected_at": str(event.get("detected_at") or ""),
                    "ended_at": str(event.get("ended_at") or ""),
                    "class_name": class_name,
                    "confidence": confidence if confidence >= 0 else None,
                    "risk_level": str(event.get("risk_level") or "none"),
                    "risk_level_name": RISK_NAMES.get(
                        str(event.get("risk_level") or "none"),
                        str(event.get("risk_level") or "none"),
                    ),
                    "detection_id": str(event.get("detection_id") or ""),
                    "alarm_id": str(event.get("alarm_id") or ""),
                    "alarm_status": alarm_status,
                    "alarm_status_name": ALARM_STATUS_NAMES.get(alarm_status, alarm_status),
                    "image_path": str(event.get("image_path") or ""),
                    "position": str(first_object.get("position") or "未知区域"),
                    "bbox_xyxy": list(first_object.get("bbox_xyxy") or event.get("bbox") or []),
                    "risk_reason": str(event_risk.get("reason") or ""),
                }
            )

        risk_reasons = self._deduplicated_text(risk_reasons)
        recommended_actions = self._deduplicated_text(recommended_actions)
        risk_level = str(task.get("highest_risk_level") or "none")
        statuses = list(alarm_status_counts)
        alarm_status = statuses[0] if len(statuses) == 1 else ("mixed" if statuses else "inactive")
        alarm_status_name = (
            ALARM_STATUS_NAMES.get(alarm_status, alarm_status)
            if alarm_status != "mixed"
            else "多种状态"
        )
        facts = {
            "monitor_source": str(task.get("display_name") or task.get("source_id") or "未知监控源"),
            "class_counts": class_counts,
            "object_count": len(report_events),
            "event_count": len(report_events),
            "max_confidence": max_confidence,
            "risk_level": risk_level,
            "risk_level_name": RISK_NAMES.get(risk_level, risk_level),
            "risk_reasons": risk_reasons,
            "recommended_actions": recommended_actions,
            "alarm_status": alarm_status,
            "alarm_status_name": alarm_status_name,
        }
        analysis, analysis_source = self._summarize_detection_facts(facts)
        return {
            "task_id": str(task.get("task_id") or ""),
            "monitor_source": facts["monitor_source"],
            "start_time": str(task.get("start_time") or ""),
            "end_time": str(task.get("end_time") or ""),
            "status": str(task.get("status") or ""),
            "event_count": len(report_events),
            "alarm_count": int(task.get("alarms_created") or 0),
            "class_counts": class_counts,
            "max_confidence": max_confidence,
            "risk_level": risk_level,
            "risk_level_name": facts["risk_level_name"],
            "alarm_status_counts": alarm_status_counts,
            "events": report_events,
            "ai_analysis": analysis,
            "analysis_source": analysis_source,
        }

    @staticmethod
    def _realtime_task_status_reply(task: Mapping[str, Any]) -> str:
        status_names = {
            "scheduled": "等待开始",
            "connecting": "正在连接",
            "running": "运行中",
            "reconnecting": "正在重连",
            "stop_requested": "正在停止",
            "completed": "已按计划完成",
            "stopped": "已人工停止",
            "failed": "执行失败",
            "interrupted": "服务重启后中断",
        }
        status = str(task.get("status") or "")
        if status in {"completed", "stopped", "failed", "interrupted"}:
            return AgentTools._realtime_terminal_summary(task)
        source_name = str(task.get("display_name") or task.get("source_id") or "当前监控")
        reply = (
            f"{source_name}实时巡检{status_names.get(status, status or '状态未知')}："
            f"已运行{float(task.get('elapsed_seconds') or 0):.1f}秒，"
            f"读取{int(task.get('frames_read') or 0)}帧，"
            f"推理{int(task.get('frames_inferred') or 0)}帧，"
            f"实际检测FPS {float(task.get('inference_fps') or 0):.2f}；"
            f"发现{int(task.get('events_detected') or 0)}个事件，"
            f"创建{int(task.get('alarms_created') or 0)}个报警，"
            f"最高风险为{RISK_NAMES.get(str(task.get('highest_risk_level') or 'none'), task.get('highest_risk_level') or 'none')}，"
            f"重连{int(task.get('reconnect_count') or 0)}次。"
        )
        error = str(task.get("last_error_message") or "").strip()
        return f"{reply}最近错误：{error}" if error else reply

    @staticmethod
    def _realtime_terminal_summary(task: Mapping[str, Any]) -> str:
        status = str(task.get("status") or "")
        if status == "completed":
            title, reason = "【实时巡检已结束】", "达到计划结束时间"
        elif status == "stopped":
            title, reason = "【实时巡检已停止】", "用户主动停止"
        else:
            title = "【实时巡检异常结束】"
            reason = str(task.get("last_error_message") or "服务重启导致任务中断")
        lines = [
            title,
            f"视频源：{task.get('display_name') or task.get('source_id') or '未知'}",
            f"任务ID：{task.get('task_id') or ''}",
            f"开始时间：{task.get('started_at') or task.get('start_time') or ''}",
            f"结束时间：{task.get('stopped_at') or task.get('end_time') or ''}",
            f"运行时长：{float(task.get('elapsed_seconds') or 0):.1f}秒",
            f"读取帧数：{int(task.get('frames_read') or 0)}",
            f"推理帧数：{int(task.get('frames_inferred') or 0)}",
            f"确认事件数：{int(task.get('events_detected') or 0)}",
            f"报警数：{int(task.get('alarms_created') or 0)}",
            f"最高风险等级：{RISK_NAMES.get(str(task.get('highest_risk_level') or 'none'), task.get('highest_risk_level') or 'none')}",
            f"结束原因：{reason}",
        ]
        if status in {"failed", "interrupted"}:
            lines.append(f"安全错误码：{task.get('last_error_code') or 'task_interrupted'}")
        return "\n".join(lines)

    def _persist_realtime_event(self, task: RealtimeInspectionTaskRecord, event: ActiveEvent) -> Mapping[str, Any]:
        from task3_alarm.alarm_rule_engine import complete_detection_alarm

        event_id = f"{task.id}-event-{event.sequence:04d}"
        existing_event = self.store.get_realtime_inspection_event(event_id)
        output_dir = OUTPUTS_DIR / "realtime_inspections" / task.source_id / task.id / "events"
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / f"event_{event.sequence:04d}.jpg"
        if existing_event is None or event.representative_updated or not image_path.is_file():
            try:
                from video_detection import _write_detection_frame
                _write_detection_frame(event.representative_frame, [event.representative_object], [], image_path)
            except Exception as exc:
                raise RealtimeInspectionError("inference_failed", "代表帧保存失败。") from exc
        image_display = self._display_path(image_path)
        start_text = event.first_seen.isoformat(sep=" ", timespec="seconds")
        end_text = event.last_seen.isoformat(sep=" ", timespec="seconds")
        obj = dict(event.representative_object)
        track_id = obj.get("track_id")
        class_counts = {event.class_name: 1}
        video_event = {
            "event_id": 1, "start_offset_seconds": 0.0,
            "end_offset_seconds": max(0.0, (event.last_seen - event.first_seen).total_seconds()),
            "start_video_time": "00:00:00.000", "end_video_time": "00:00:00.000",
            "start_real_time": start_text, "end_real_time": end_text,
            "object_count": 1, "max_simultaneous_objects": 1, "unique_object_count": 1,
            "class_counts": class_counts, "classes": [event.class_name], "observed_classes": [event.class_name],
            "max_confidence": event.confidence, "positive_sample_count": event.hit_count,
            "track_ids": [track_id] if track_id is not None else [],
            "tracks": [{"track_id": track_id, "class": event.class_key, "class_name": event.class_name,
                        "max_confidence": event.confidence, "confirmed_observation_count": event.hit_count,
                        "representative_frame": image_display, "representative_object": obj}],
            "key_frame": image_display,
            "key_frames": [{"image": image_display, "real_time": start_text, "object_count": 1,
                            "class_counts": class_counts, "track_ids": [track_id] if track_id is not None else []}],
            "frame_images": [image_display],
        }
        existing_detection = (
            self.store.get_detection(existing_event.detection_id)
            if existing_event is not None else None
        )
        created_at = str(
            (existing_detection.summary.get("created_at") if existing_detection else "")
            or self.current_time().isoformat(sep=" ", timespec="seconds")
        )
        detection = {
            "status": "completed", "created_at": created_at,
            "video": f"realtime://{task.source_id}/{task.id}", "video_start_time": start_text,
            "video_end_time": end_text, "duration_seconds": video_event["end_offset_seconds"],
            "source_fps": None, "requested_sample_fps": task.sample_fps, "sample_fps": task.sample_fps,
            "sampled_frames": event.hit_count, "positive_frames": event.hit_count,
            "num_detection_boxes": event.hit_count, "unique_object_count": 1,
            "has_foreign_object": True, "num_events": 1, "class_counts": class_counts,
            "events": [video_event], "tracks": video_event["tracks"], "detection_frames": [],
            "thresholds": task.config.get("parameters") or {},
        }
        result_json = output_dir / f"event_{event.sequence:04d}.json"
        alarm_json = output_dir / f"event_{event.sequence:04d}_alarm.json"
        report_path = output_dir / f"event_{event.sequence:04d}_report.txt"
        result_json.write_text(json.dumps(detection, ensure_ascii=False, indent=2), encoding="utf-8")
        alarm_document, alarm_report = complete_detection_alarm(
            detection, input_json=result_json, output_json=alarm_json, output_txt=report_path, source_type="video"
        )
        if existing_event is None:
            detection_record, alarm_record = self.store.record_detection(
                task.session_id, source_type="realtime", source_path=f"realtime://{task.source_id}/{task.id}",
                detection=detection, alarm_document=alarm_document, alarm_report=alarm_report,
                line_id=task.line_id, source_started_at=start_text, source_ended_at=end_text,
            )
        else:
            alarm_document["report_id"] = existing_event.alarm_id
            alarm_json.write_text(
                json.dumps(alarm_document, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            detection_record, alarm_record = self.store.update_detection_alarm(
                existing_event.detection_id, existing_event.alarm_id,
                detection=detection, alarm_document=alarm_document,
                alarm_report=alarm_report, source_ended_at=end_text,
            )
        generated = alarm_document.get("generated_report") or {}
        document_events = [
            item for item in alarm_document.get("events") or [] if isinstance(item, Mapping)
        ]
        risk_reasons = self._deduplicated_text(
            [
                ((document_events[0].get("risk") or {}).get("reason") if document_events else ""),
                (alarm_document.get("overall_risk") or {}).get("reason"),
            ]
        )
        recommended_actions = self._deduplicated_text(
            [generated.get("recommended_action")]
        )
        summary_facts = {
            "monitor_source": task.source_id,
            "class_counts": class_counts,
            "object_count": 1,
            "max_confidence": event.confidence,
            "risk_level": alarm_record.risk_level,
            "risk_level_name": RISK_NAMES.get(alarm_record.risk_level, alarm_record.risk_level),
            "risk_reasons": risk_reasons,
            "recommended_actions": recommended_actions,
            "alarm_status": alarm_record.status,
            "alarm_status_name": ALARM_STATUS_NAMES.get(alarm_record.status, alarm_record.status),
        }
        llm_summary = (
            existing_event.llm_summary
            if existing_event is not None
            else self._fallback_summary(summary_facts)
        )
        metadata = {"event_id": event_id, "task_id": task.id, "source_id": task.source_id,
                    "line_id": task.line_id, "detected_at": start_text, "ended_at": end_text,
                    "last_seen_at": end_text, "event_status": event.event_status,
                    "class_name": event.class_name, "confidence": event.confidence, "bbox": event.bbox,
                    "hit_count": event.hit_count, "risk_level": alarm_record.risk_level,
                    "class_counts": class_counts, "max_confidence": event.confidence,
                    "detection_id": detection_record.id, "alarm_id": alarm_record.id,
                    "alarm_status": alarm_record.status, "image_path": image_display,
                    "llm_summary": llm_summary, "analysis_source": (
                        str(existing_event.metadata.get("analysis_source") or "fallback")
                        if existing_event is not None else "fallback"
                    ), "alarm_json_path": self._display_path(alarm_json),
                    "alarm_report_path": self._display_path(report_path)}
        result_json.write_text(json.dumps({**detection, **metadata}, ensure_ascii=False, indent=2), encoding="utf-8")
        if existing_event is None:
            self.store.record_realtime_inspection_event(
                event_id=event_id, task_id=task.id, source_id=task.source_id,
                detected_at=start_text, ended_at=end_text, last_seen_at=end_text,
                event_status=event.event_status, class_name=event.class_name,
                confidence=event.confidence, max_confidence=event.confidence,
                bbox=event.bbox, risk_level=alarm_record.risk_level,
                detection_id=detection_record.id, alarm_id=alarm_record.id, image_path=image_display,
                metadata=metadata, line_id=task.line_id, hit_count=event.hit_count,
                class_counts=class_counts, alarm_report=alarm_report, llm_summary=llm_summary,
            )
        else:
            self.store.update_realtime_inspection_event(
                event_id, ended_at=end_text, last_seen_at=end_text,
                event_status=event.event_status, class_name=event.class_name,
                confidence=event.confidence, max_confidence=event.confidence,
                bbox=event.bbox, risk_level=alarm_record.risk_level,
                image_path=image_display, hit_count=event.hit_count,
                class_counts=class_counts, metadata=metadata,
                alarm_report=alarm_report, llm_summary=llm_summary,
            )
        if existing_event is None:
            self._schedule_realtime_event_summary(event_id, summary_facts)
        return metadata

    def control_stream_archive(
        self,
        session_id: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        del session_id
        action = str(context.get("action") or "query").strip().lower()
        source_id = str(context.get("source_id") or "").strip().lower()
        try:
            registry = self._video_source_registry_loader()
            source = registry.get(source_id)
        except (FileNotFoundError, ValueError):
            return {
                "ok": False,
                "error_code": "configuration_error",
                "reply": "视频源注册表尚未正确配置。",
                "data": {},
            }
        except LookupError:
            return {
                "ok": False,
                "error_code": "source_not_found",
                "reply": f"未找到已注册的视频源：{source_id}",
                "data": {},
            }
        if not source.is_rtsp or source.stream is None:
            return {
                "ok": False,
                "error_code": "not_rtsp_source",
                "reply": f"{source.display_name}不是 RTSP 视频源，无法持续归档。",
                "data": {},
            }

        if action == "start":
            segment_seconds = float(
                context.get("segment_seconds") or source.stream.segment_seconds
            )
            retention_seconds = float(context.get("retention_hours", 24.0)) * 3600.0
            try:
                state = self._archive_manager.start(
                    source.source_id,
                    segment_seconds=segment_seconds,
                    retention_seconds=retention_seconds,
                )
            except ValueError as exc:
                return {
                    "ok": False,
                    "error_code": "invalid_archive_config",
                    "reply": str(exc),
                    "data": {},
                }
            return {
                "ok": True,
                "reply": (
                    f"已启动{source.display_name}录像归档：每 {segment_seconds:g} 秒保存一段，"
                    f"保留 {retention_seconds / 3600:g} 小时。"
                ),
                "data": {
                    **state.to_dict(),
                    "display_name": source.display_name,
                    "line_id": source.line_id,
                    "manifest_path": self._display_path(
                        self._archive_manager.manifest_path(source.source_id)
                    ),
                },
            }

        if action == "stop":
            try:
                state = self._archive_manager.stop(source.source_id)
            except LookupError as exc:
                return {
                    "ok": False,
                    "error_code": "archive_not_found",
                    "reply": str(exc),
                    "data": {},
                }
            return {
                "ok": True,
                "reply": f"已请求停止{source.display_name}录像归档；当前片段完成后停止。",
                "data": state.to_dict(),
            }

        state = self.store.get_stream_archive_state(source.source_id)
        if state is None:
            return {
                "ok": True,
                "reply": f"{source.display_name}尚未启动录像归档。",
                "data": {
                    "found": False,
                    "source_id": source.source_id,
                    "display_name": source.display_name,
                },
            }
        limit = int(context.get("limit", 100))
        segments = self.store.list_stream_archive_segments(
            source.source_id,
            statuses=("ready", "failed"),
            limit=limit,
        )
        return {
            "ok": True,
            "reply": (
                f"{source.display_name}录像归档状态为 {state.status}，"
                f"当前索引返回 {len(segments)} 个片段。"
            ),
            "data": {
                "found": True,
                **state.to_dict(),
                "display_name": source.display_name,
                "line_id": source.line_id,
                "manifest_path": self._display_path(
                    self._archive_manager.manifest_path(source.source_id)
                ),
                "segments": [
                    {
                        **segment.to_dict(),
                        "video_path": self._display_path(Path(segment.video_path)),
                    }
                    for segment in segments
                ],
            },
        }

    def detect_archived_video(
        self,
        session_id: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        source_id = str(context.get("source_id") or "").strip().lower()
        try:
            registry = self._video_source_registry_loader()
            source = registry.get(source_id)
        except (FileNotFoundError, ValueError):
            return {
                "ok": False,
                "error_code": "configuration_error",
                "reply": "视频源注册表尚未正确配置。",
                "data": {},
            }
        except LookupError:
            return {
                "ok": False,
                "error_code": "source_not_found",
                "reply": f"未找到已注册的视频源：{source_id}",
                "data": {},
            }

        start = datetime.fromisoformat(
            str(context["start_time"]).replace("Z", "+00:00")
        )
        end = datetime.fromisoformat(str(context["end_time"]).replace("Z", "+00:00"))
        if end > self.current_time():
            return {
                "ok": False,
                "error_code": "archive_range_in_future",
                "reply": "历史录像检测的结束时间不能晚于当前时间；实时画面请使用实时检测或监控任务。",
                "data": {
                    "requested_range": {
                        "start_time": start.isoformat(),
                        "end_time": end.isoformat(),
                    }
                },
            }

        parameters = dict(context.get("parameters") or {})
        zone_id = str(context.get("zone_id") or "").strip().lower()
        selected_zone: Dict[str, Any] = {}
        if zone_id:
            zone = next((item for item in source.zones if item.zone_id == zone_id), None)
            if zone is None:
                return {
                    "ok": False,
                    "error_code": "zone_not_found",
                    "reply": f"视频源 {source.display_name} 未注册区域：{zone_id}",
                    "data": {"available_zones": [item.zone_id for item in source.zones]},
                }
            parameters["roi"] = list(zone.roi)
            selected_zone = {
                "zone_id": zone.zone_id,
                "display_name": zone.display_name,
                "roi": list(zone.roi),
            }

        coverage = self._archive_manager.resolve_range(
            source.source_id,
            start_time=start,
            end_time=end,
            tolerance_seconds=float(context.get("coverage_tolerance_seconds", 2.0)),
        )
        if coverage.missing_segments:
            return {
                "ok": False,
                "error_code": "archive_segment_missing",
                "reply": "历史录像索引存在，但部分录像文件已缺失，已拒绝用实时画面替代。",
                "data": coverage.to_dict(),
            }
        if coverage.gaps:
            return {
                "ok": False,
                "error_code": "archive_coverage_gap",
                "reply": "请求时段的历史录像覆盖不完整，已拒绝用实时画面替代。",
                "data": coverage.to_dict(),
            }

        segment_results: list[Dict[str, Any]] = []
        class_counts: Dict[str, int] = {}
        event_frames: list[Dict[str, Any]] = []
        reports: list[str] = []
        risk_rank = {"none": 0, "low": 1, "medium": 2, "high": 3}
        overall_risk = "none"
        total_events = 0
        for segment in coverage.segments:
            segment_start = datetime.fromisoformat(segment.started_at)
            segment_end = datetime.fromisoformat(segment.ended_at)
            start_offset = max(0.0, (start - segment_start).total_seconds())
            clipped_end = min(end, segment_end)
            end_offset = max(start_offset, (clipped_end - segment_start).total_seconds())
            video_path = Path(segment.video_path)
            if not video_path.is_absolute():
                video_path = PROJECT_ROOT / video_path
            detection_context: Dict[str, Any] = {
                "video_path": str(video_path),
                "video_start_time": segment.started_at,
                "source_ended_at": clipped_end.isoformat(timespec="seconds"),
                "line_id": source.line_id,
                "parameters": parameters,
            }
            if start_offset > 0.001:
                detection_context["start_offset_seconds"] = start_offset
            if clipped_end < segment_end - timedelta(milliseconds=1):
                detection_context["end_offset_seconds"] = end_offset
            result = self.detect_video(session_id, detection_context)
            data = dict(result.get("data") or {})
            segment_results.append(
                {
                    "segment_id": segment.segment_id,
                    "started_at": segment.started_at,
                    "ended_at": segment.ended_at,
                    "start_offset_seconds": start_offset,
                    "end_offset_seconds": end_offset,
                    **result,
                }
            )
            if not result.get("ok"):
                return {
                    "ok": False,
                    "error_code": str(result.get("error_code") or "historical_detection_failed"),
                    "reply": result.get("reply") or "历史录像检测执行失败。",
                    "data": {
                        **coverage.to_dict(),
                        "segment_results": segment_results,
                    },
                }
            total_events += int(data.get("event_count") or 0)
            risk = str(data.get("risk_level") or "none")
            if risk_rank.get(risk, 0) > risk_rank.get(overall_risk, 0):
                overall_risk = risk
            for name, count in dict(data.get("class_counts") or {}).items():
                class_counts[str(name)] = class_counts.get(str(name), 0) + int(count)
            for frame in data.get("event_frames") or []:
                if isinstance(frame, Mapping):
                    event_frames.append(dict(frame))
            report = str(data.get("alarm_report") or "").strip()
            if report:
                reports.append(report)

        return {
            "ok": True,
            "reply": (
                f"{source.display_name}历史录像检测完成：覆盖 {len(coverage.segments)} 个录像片段，"
                f"发现 {total_events} 个异物事件，总体为{RISK_NAMES.get(overall_risk, overall_risk)}。"
            ),
            "data": {
                **coverage.to_dict(),
                "display_name": source.display_name,
                "line_id": source.line_id,
                "zone": selected_zone,
                "risk_level": overall_risk,
                "event_count": total_events,
                "class_counts": class_counts,
                "alarm_report": "\n\n".join(reports),
                "event_frames": event_frames,
                "segment_results": segment_results,
                "workflow": [
                    "lookup-archive-segments",
                    "verify-archive-coverage",
                    "detect-video",
                    "assess-risk",
                    "persist-history",
                    "create-alarm",
                ],
            },
        }

    @staticmethod
    def video_event_frames(detection: Dict[str, Any]) -> list[Dict[str, Any]]:
        frames: list[Dict[str, Any]] = []
        for index, event in enumerate(detection.get("events") or [], start=1):
            if not isinstance(event, dict):
                continue
            key_frame = str(event.get("key_frame") or "")
            if not key_frame:
                continue
            frames.append(
                {
                    "event_id": _integer(event.get("event_id"), index),
                    "key_frame": key_frame,
                }
            )
        return frames

    @staticmethod
    def image_event_frames(
        detection: Dict[str, Any], visualization_image: str = ""
    ) -> list[Dict[str, Any]]:
        key_frame = str(visualization_image or "")
        if not key_frame:
            visualization_dir = str(detection.get("visualization_dir") or "")
            source = str(detection.get("source") or "")
            if visualization_dir and source:
                key_frame = str(Path(visualization_dir) / Path(source).name)
        return [{"event_id": 1, "key_frame": key_frame}] if key_frame else []

    def detect_image(self, session_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        raw_path = context.get("image_path")
        if not raw_path:
            return {
                "ok": False,
                "requires_attachment": True,
                "reply": "还没有看到图片/视频，发过来立刻帮你分析。",
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
        with self._detection_lock:
            outcome = self._image_detection_runner(image_path, parameters)
        stored_detection = dict(outcome.detection)
        if outcome.visualization_image:
            stored_detection["visualization_image"] = outcome.visualization_image
        if outcome.visualization_dir:
            stored_detection["visualization_dir"] = outcome.visualization_dir
        detection_record, alarm_record = self.store.record_detection(
            session_id,
            source_type="image",
            source_path=str(image_path),
            detection=stored_detection,
            alarm_document=outcome.alarm_document,
            alarm_report=outcome.alarm_report,
            line_id=str(context.get("line_id") or ""),
            source_started_at=_normalize_datetime_text(
                context.get("source_started_at") or context.get("captured_at")
            ),
            source_ended_at=_normalize_datetime_text(
                context.get("source_ended_at") or context.get("captured_at")
            ),
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
                "alarm_report": outcome.alarm_report,
                "visualization_dir": outcome.visualization_dir,
                "visualization_image": outcome.visualization_image,
                "event_frames": self.image_event_frames(
                    outcome.detection, outcome.visualization_image
                ),
            },
        }

    def detect_video(self, session_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        raw_path = context.get("video_path")
        if not raw_path:
            return {
                "ok": False,
                "requires_attachment": True,
                "reply": "还没有看到图片/视频，发过来立刻帮你分析。",
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

        start_offset = _float(context.get("start_offset_seconds"), 0.0)
        end_offset = (
            _float(context.get("end_offset_seconds"))
            if context.get("end_offset_seconds") is not None
            else None
        )
        detection_video_path = video_path
        if start_offset > 0 or end_offset is not None:
            detection_video_path = self._video_segmenter(
                video_path,
                start_offset,
                end_offset,
            )
            video_start = video_start + timedelta(seconds=start_offset)

        parameters = dict(context.get("parameters") or {})
        with self._detection_lock:
            outcome = self._detection_runner(detection_video_path, video_start, parameters)
        source_started_at = _normalize_datetime_text(
            outcome.detection.get("video_start_time") or video_start
        )
        source_ended_at = _normalize_datetime_text(
            outcome.detection.get("video_end_time")
            or context.get("source_ended_at")
            or (
                video_start
                + timedelta(seconds=_float(outcome.detection.get("duration_seconds")))
            )
        )
        detection_record, alarm_record = self.store.record_detection(
            session_id,
            source_type="video",
            source_path=str(video_path),
            detection=outcome.detection,
            alarm_document=outcome.alarm_document,
            alarm_report=outcome.alarm_report,
            line_id=str(context.get("line_id") or ""),
            source_started_at=source_started_at,
            source_ended_at=source_ended_at,
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
                f"总体为{risk_name}。"
            ),
            "data": {
                "detection_id": detection_record.id,
                "alarm_id": alarm_record.id,
                "risk_level": alarm_record.risk_level,
                "alarm_status": alarm_record.status,
                "event_count": event_count,
                "class_counts": outcome.detection.get("class_counts") or {},
                "line_id": detection_record.line_id,
                "source_started_at": detection_record.source_started_at,
                "source_ended_at": detection_record.source_ended_at,
                "start_offset_seconds": start_offset,
                "end_offset_seconds": end_offset,
                "segment_path": (
                    self._display_path(detection_video_path)
                    if detection_video_path != video_path
                    else ""
                ),
                "result_json": outcome.result_json,
                "alarm_json": outcome.alarm_json,
                "alarm_report_path": outcome.alarm_report_path,
                "alarm_report": outcome.alarm_report,
                "event_frames": self.video_event_frames(outcome.detection),
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
                "alarm_report": record.alarm_report,
                "event_frames": (
                    self.video_event_frames(record.summary)
                    if record.source_type in {"video", "realtime"}
                    else self.image_event_frames(record.summary)
                ),
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

    def assess_risk(self, session_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        del session_id
        raw_json = context.get("detection_json")
        detection = context.get("detection")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = OUTPUTS_DIR / "agent_risk_assessments" / timestamp
        if raw_json:
            input_json = Path(str(raw_json)).expanduser()
            if not input_json.is_file():
                return {
                    "ok": False,
                    "reply": f"找不到检测结果 JSON：{input_json}",
                    "data": {"detection_json": str(input_json)},
                }
            detection = json.loads(input_json.read_text(encoding="utf-8"))
        elif isinstance(detection, dict):
            output_dir.mkdir(parents=True, exist_ok=True)
            input_json = output_dir / "detection_input.json"
            input_json.write_text(
                json.dumps(detection, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            return {
                "ok": False,
                "reply": "请提供 detection_json 路径或 detection 对象。",
                "data": {},
            }

        output_dir.mkdir(parents=True, exist_ok=True)
        alarm_json = output_dir / "unified_alarm.json"
        alarm_report_path = output_dir / "alarm_report.txt"
        from task3_alarm.alarm_rule_engine import complete_detection_alarm

        alarm_document, alarm_report = complete_detection_alarm(
            detection,
            input_json=input_json,
            output_json=alarm_json,
            output_txt=alarm_report_path,
            source_type=str(context.get("source_type") or "auto"),
        )
        overall = alarm_document.get("overall_risk") or {}
        generated = alarm_document.get("generated_report") or {}
        risk_level = str(overall.get("level") or "none")
        return {
            "ok": True,
            "reply": (
                f"风险研判完成：{RISK_NAMES.get(risk_level, risk_level)}。"
                f"处置建议：{generated.get('recommended_action') or '继续监测。'}"
            ),
            "data": {
                "risk_level": risk_level,
                "requires_stop": bool(overall.get("requires_stop")),
                "reason": overall.get("reason") or "",
                "recommended_action": generated.get("recommended_action") or "",
                "alarm_document": alarm_document,
                "alarm_report": alarm_report,
                "alarm_json": self._display_path(alarm_json),
                "alarm_report_path": self._display_path(alarm_report_path),
            },
        }

    def parse_detection_result(
        self, session_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        del session_id
        raw_json = context.get("detection_json")
        detection = context.get("detection")
        if raw_json:
            input_json = Path(str(raw_json)).expanduser()
            if not input_json.is_file():
                return {
                    "ok": False,
                    "reply": f"找不到检测结果 JSON：{input_json}",
                    "data": {"detection_json": str(input_json)},
                }
            detection = json.loads(input_json.read_text(encoding="utf-8"))
        elif isinstance(detection, dict):
            input_json = PROJECT_ROOT / "in_memory_detection.json"
        else:
            return {
                "ok": False,
                "reply": "请提供 detection_json 路径或 detection 对象。",
                "data": {},
            }

        from task3_alarm.unified_alarm import convert_detection

        normalized = convert_detection(
            detection,
            input_json=input_json,
            source_type=str(context.get("source_type") or "auto"),
        )
        source = normalized.get("source") or {}
        summary = normalized.get("detection_summary") or {}
        events = normalized.get("events") or []
        return {
            "ok": True,
            "reply": (
                f"检测结果解析完成：来源 {source.get('type') or 'unknown'}，"
                f"共 {len(events)} 个事件。"
            ),
            "data": {
                "source": source,
                "detection_summary": summary,
                "events": events,
                "event_count": len(events),
                "normalized_detection": normalized,
                "candidate_count": _integer(detection.get("num_candidates")),
                "candidates": detection.get("candidates")
                or detection.get("unknown_candidates")
                or [],
            },
        }

    def current_alarm(self, session_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        alarm = self.store.current_alarm(
            session_id if context.get("session_only", False) else None,
            line_id=str(context.get("line_id") or ""),
        )
        if alarm is None:
            return {
                "ok": True,
                "reply": "当前没有可查看的报警。",
                "data": {"found": False},
            }
        return {
            "ok": True,
            "reply": (
                f"当前报警 {alarm.id}：{RISK_NAMES.get(alarm.risk_level, alarm.risk_level)}，"
                f"状态 {alarm.status}。"
            ),
            "data": {
                "found": True,
                "alarm_id": alarm.id,
                "detection_id": alarm.detection_id,
                "risk_level": alarm.risk_level,
                "alarm_status": alarm.status,
                "requires_stop": alarm.requires_stop,
                "report": alarm.report,
                "report_text": alarm.report_text,
                "created_at": alarm.created_at,
                "updated_at": alarm.updated_at,
            },
        }

    def query_history(self, session_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        del session_id
        records = self.store.query_detections(
            start_time=_normalize_datetime_text(context.get("start_time")),
            end_time=_normalize_datetime_text(context.get("end_time")),
            risk_level=str(context.get("risk_level") or ""),
            line_id=str(context.get("line_id") or ""),
            source_type=str(context.get("source_type") or ""),
            review_status=str(context.get("review_status") or ""),
            limit=_integer(context.get("limit"), 100),
        )
        include_details = bool(context.get("include_details", True))
        items = []
        for record in records:
            item = {
                "detection_id": record.id,
                "source_type": record.source_type,
                "source_path": record.source_path,
                "line_id": record.line_id,
                "source_started_at": record.source_started_at,
                "source_ended_at": record.source_ended_at,
                "risk_level": record.risk_level,
                "status": record.status,
                "review_status": record.review_status,
                "review_note": record.review_note,
                "reviewer": record.reviewer,
                "reviewed_at": record.reviewed_at,
                "class_counts": record.summary.get("class_counts") or {},
                "created_at": record.created_at,
            }
            if include_details:
                item["detection"] = record.summary
                item["alarm_report"] = record.alarm_report
            items.append(item)
        return {
            "ok": True,
            "reply": f"共查询到 {len(items)} 条检测记录。",
            "data": {"count": len(items), "records": items},
        }

    def generate_risk_report(
        self, session_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        del session_id
        target_date = context.get("date")
        start_time = context.get("start_time")
        end_time = context.get("end_time")
        if target_date and not start_time and not end_time:
            day = target_date if isinstance(target_date, date) else date.fromisoformat(str(target_date))
            timezone = self._now().tzinfo
            start_time = datetime.combine(day, time.min, tzinfo=timezone)
            end_time = datetime.combine(day, time.max, tzinfo=timezone)
        summary = self.store.filtered_summary(
            start_time=_normalize_datetime_text(start_time),
            end_time=_normalize_datetime_text(end_time),
            risk_level=str(context.get("risk_level") or ""),
            line_id=str(context.get("line_id") or ""),
            source_type=str(context.get("source_type") or ""),
            review_status=str(context.get("review_status") or ""),
        )
        risks = summary["risk_counts"]
        alarm_statuses = summary["alarm_status_counts"]
        classes = summary["class_counts"]
        class_text = "、".join(f"{name}{count}个" for name, count in classes.items()) or "无"
        report = (
            "风险汇总报告\n"
            f"- 时间范围：{summary['start_time'] or '不限'} 至 {summary['end_time'] or '不限'}\n"
            f"- 线路：{summary['line_id'] or '全部'}\n"
            f"- 检测轮次：{summary['detection_count']}\n"
            f"- 高/中/低风险：{risks.get('high', 0)}/{risks.get('medium', 0)}/{risks.get('low', 0)}\n"
            f"- 待确认/已确认/已取消报警：{alarm_statuses.get('pending', 0)}/"
            f"{alarm_statuses.get('confirmed', 0)}/{alarm_statuses.get('cancelled', 0)}\n"
            f"- 异物统计：{class_text}"
        )
        return {"ok": True, "reply": report, "data": {**summary, "report": report}}

    def review_detection(self, session_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        detection_id = str(context.get("detection_id") or "").strip()
        if not detection_id:
            latest = self.store.latest_detection(session_id)
            detection_id = latest.id if latest else ""
        if not detection_id:
            return {
                "ok": False,
                "reply": "没有找到可复核的检测记录。",
                "data": {"found": False},
            }
        updated = self.store.set_detection_review(
            detection_id,
            session_id,
            str(context.get("action") or "confirm"),
            reviewer=str(context.get("reviewer") or ""),
            note=str(context.get("note") or ""),
        )
        return {
            "ok": True,
            "reply": f"检测记录 {updated.id} 已更新为 {updated.review_status}。",
            "data": {
                "found": True,
                "detection_id": updated.id,
                "review_status": updated.review_status,
                "reviewer": updated.reviewer,
                "review_note": updated.review_note,
                "reviewed_at": updated.reviewed_at,
            },
        }

    def confirm_alarm(self, session_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return self._control_alarm("confirm", session_id, context)

    def cancel_alarm(self, session_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return self._control_alarm("cancel", session_id, context)

    def _control_alarm(
        self, action: str, session_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        scope = str(context.get("scope") or "single").strip().lower()
        if scope == "realtime_task":
            return self._control_realtime_task_alarms(action, session_id, context)
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
        updated = self.store.set_alarm_action(
            alarm.id,
            session_id,
            action,
            note=str(context.get("note") or ""),
        )
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

    def _control_realtime_task_alarms(
        self, action: str, session_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        task_id = str(context.get("task_id") or "").strip().lower()
        task = self.store.get_realtime_inspection_task(task_id) if task_id else None
        if task is None and not task_id:
            recent = self.store.list_realtime_inspection_tasks(
                session_id=session_id, limit=1,
            )
            task = recent[0] if recent else None
        if task is None or task.session_id != session_id:
            return {
                "ok": False,
                "reply": "当前会话找不到可闭环的实时巡检任务。",
                "data": {"found": False, "scope": "realtime_task"},
            }

        events = self.store.list_realtime_inspection_events(task.id, limit=None)
        alarm_ids = list(dict.fromkeys(
            str(event.alarm_id).strip() for event in events if str(event.alarm_id).strip()
        ))
        if not alarm_ids:
            return {
                "ok": True,
                "reply": "本轮实时巡检没有产生需要确认或取消的报警。",
                "data": {
                    "found": True,
                    "scope": "realtime_task",
                    "task_id": task.id,
                    "affected_count": 0,
                    "unchanged_count": 0,
                    "skipped_count": 0,
                },
            }

        target_status = "confirmed" if action == "confirm" else "cancelled"
        affected_ids: list[str] = []
        unchanged_ids: list[str] = []
        skipped_ids: list[str] = []
        for alarm_id in alarm_ids:
            alarm = self.store.get_alarm(alarm_id)
            if alarm is None or alarm.session_id != session_id or alarm.status == "inactive":
                skipped_ids.append(alarm_id)
                continue
            if alarm.status == target_status:
                unchanged_ids.append(alarm_id)
                continue
            if self._alarm_control_handler is not None:
                self._alarm_control_handler(action, alarm)
            self.store.set_alarm_action(
                alarm.id,
                session_id,
                action,
                note=str(context.get("note") or ""),
            )
            affected_ids.append(alarm.id)

        action_text = "确认" if action == "confirm" else "取消"
        reply = f"已{action_text}本轮实时巡检的{len(affected_ids)}条报警"
        details = []
        if unchanged_ids:
            details.append(f"{len(unchanged_ids)}条此前已是该状态")
        if skipped_ids:
            details.append(f"{len(skipped_ids)}条不可操作或记录缺失")
        if details:
            reply += "；" + "，".join(details)
        reply += "。"
        return {
            "ok": True,
            "reply": reply,
            "data": {
                "found": True,
                "scope": "realtime_task",
                "task_id": task.id,
                "action": action,
                "alarm_status": target_status,
                "affected_count": len(affected_ids),
                "unchanged_count": len(unchanged_ids),
                "skipped_count": len(skipped_ids),
                "alarm_ids": affected_ids,
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
    def _extract_video_segment(
        video_path: Path,
        start_offset: float,
        end_offset: Optional[float],
    ) -> Path:
        """Extract a detection-only MP4 segment without changing detector internals."""
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"OpenCV 无法打开待切片视频：{video_path}")
        writer = None
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
                raise ValueError("无法读取视频 FPS、帧数或画面尺寸，不能按时间切片")
            duration = frame_count / fps
            effective_end = duration if end_offset is None else float(end_offset)
            if start_offset >= duration:
                raise ValueError(
                    f"start_offset {start_offset:g} 超出视频时长 {duration:.3f} 秒"
                )
            if effective_end > duration + (0.5 / fps):
                raise ValueError(
                    f"end_offset {effective_end:g} 超出视频时长 {duration:.3f} 秒"
                )

            start_frame = max(0, int(round(start_offset * fps)))
            end_frame = min(frame_count, int(round(effective_end * fps)))
            if end_frame <= start_frame:
                raise ValueError("视频切片结束帧必须晚于开始帧")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            segment_dir = OUTPUTS_DIR / "agent_video_segments"
            segment_dir.mkdir(parents=True, exist_ok=True)
            segment_path = segment_dir / (
                f"{video_path.stem}_{start_offset:g}_{effective_end:g}_{timestamp}.mp4"
            )
            writer = cv2.VideoWriter(
                str(segment_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                raise ValueError("无法创建 MP4 视频切片")

            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(start_frame, end_frame):
                ok, frame = capture.read()
                if not ok:
                    break
                writer.write(frame)
        finally:
            capture.release()
            if writer is not None:
                writer.release()
        if not segment_path.is_file() or segment_path.stat().st_size == 0:
            raise ValueError("视频切片生成失败")
        return segment_path

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
