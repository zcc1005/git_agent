from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path
import re
from typing import Any, Dict, Mapping

from agent.tools import AgentTools

from .base import RuntimeSkill, SkillRegistry, SkillSpec
from .schemas import ALL_SKILL_SCHEMAS


RISK_LEVELS = {"none", "low", "medium", "high"}
SOURCE_TYPES = {"image", "video", "realtime"}
REVIEW_STATUSES = {"unreviewed", "confirmed", "rejected", "closed"}
ALARM_ACTIONS = {"query", "confirm", "cancel"}
ALARM_READ_ACTION_ALIASES = {
    "view": "query",
    "show": "query",
    "get": "query",
    "status": "query",
}
IMAGE_PARAMETERS = {
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
VIDEO_PARAMETERS = {
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
VIDEO_SOURCE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")


def _parameters(values: Dict[str, Any], allowed: set[str]) -> Dict[str, Any]:
    parameters = values.get("parameters") or {}
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters 必须是对象")
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        raise ValueError(f"不支持的检测参数：{', '.join(unknown)}")
    normalized = dict(parameters)
    conf = float(normalized.get("conf", 0.25))
    known_conf = float(normalized.get("known_conf", 0.40))
    if not 0 <= conf < known_conf <= 1:
        raise ValueError("阈值必须满足 0 <= conf < known_conf <= 1")
    if "imgsz" in normalized and int(normalized["imgsz"]) < 32:
        raise ValueError("imgsz 必须不小于 32")
    for name in (
        "nms_iou",
        "duplicate_iou",
        "duplicate_containment",
        "cross_class_iou",
        "cross_class_containment",
        "max_area_ratio",
        "track_iou",
    ):
        if name in normalized and not 0 < float(normalized[name]) < 1:
            raise ValueError(f"{name} 必须位于 0 到 1 之间")
    values["parameters"] = normalized
    return values


def _validate_image(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    return _parameters(dict(arguments), IMAGE_PARAMETERS)


def _validate_video(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    values = _parameters(dict(arguments), VIDEO_PARAMETERS)
    parameters = values["parameters"]
    if "sample_fps" in parameters and not 0 < float(parameters["sample_fps"]) <= 60:
        raise ValueError("sample_fps 必须大于 0 且不超过 60")
    if float(parameters.get("event_silence_seconds", 1.0)) <= 0:
        raise ValueError("event_silence_seconds 必须大于 0")
    if float(parameters.get("track_max_age_seconds", 1.0)) < 0:
        raise ValueError("track_max_age_seconds 不能小于 0")
    if int(parameters.get("min_unknown_hits", 2)) < 1:
        raise ValueError("min_unknown_hits 必须至少为 1")
    if float(parameters.get("track_center_distance_ratio", 3.0)) <= 0:
        raise ValueError("track_center_distance_ratio 必须大于 0")
    conf = float(parameters.get("conf", 0.25))
    known_conf = float(parameters.get("known_conf", 0.40))
    unknown_conf = float(parameters.get("unknown_single_frame_conf", 0.40))
    if not conf <= unknown_conf <= known_conf:
        raise ValueError("unknown_single_frame_conf 必须位于 conf 与 known_conf 之间")
    start_offset = float(values.get("start_offset_seconds", 0.0))
    end_offset = values.get("end_offset_seconds")
    if start_offset < 0:
        raise ValueError("start_offset_seconds 不能小于 0")
    if end_offset is not None and float(end_offset) <= start_offset:
        raise ValueError("end_offset_seconds 必须大于 start_offset_seconds")
    values["start_offset_seconds"] = start_offset
    if end_offset is not None:
        values["end_offset_seconds"] = float(end_offset)
    if "roi" in parameters and parameters["roi"] is not None:
        roi = parameters["roi"]
        if not isinstance(roi, (list, tuple)) or len(roi) != 4:
            raise ValueError("roi 必须为 [x1, y1, x2, y2]")
        x1, y1, x2, y2 = (int(value) for value in roi)
        if min(x1, y1) < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("roi 坐标必须非负，且 x2>x1、y2>y1")
        parameters["roi"] = (x1, y1, x2, y2)
    return values


def _validate_risk(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    values = dict(arguments)
    if values.get("detection_json") and isinstance(values.get("detection"), dict):
        raise ValueError("detection_json 与 detection 只能提供一个")
    if not values.get("detection_json") and not isinstance(values.get("detection"), dict):
        raise ValueError("必须提供 detection_json 或 detection")
    source_type = str(values.get("source_type") or "auto").lower()
    if source_type not in SOURCE_TYPES | {"auto"}:
        raise ValueError("source_type 只能是 auto、image 或 video")
    values["source_type"] = source_type
    return values


def _validate_alarm(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    values = dict(arguments)
    action = str(values.get("action") or "query").strip().lower()
    action = ALARM_READ_ACTION_ALIASES.get(action, action)
    if action not in ALARM_ACTIONS:
        raise ValueError("action 只能是 query、confirm 或 cancel")
    scope = str(values.get("scope") or "single").strip().lower()
    if scope not in {"single", "realtime_task"}:
        raise ValueError("scope 只能是 single 或 realtime_task")
    if scope == "realtime_task" and action == "query":
        raise ValueError("realtime_task scope 只用于明确的 confirm 或 cancel 写操作")
    if scope == "realtime_task" and values.get("alarm_id"):
        raise ValueError("批量闭环不能同时指定 alarm_id")
    values["action"] = action
    values["scope"] = scope
    return values


def _validate_filters(
    arguments: Mapping[str, Any], *, include_limit: bool
) -> Dict[str, Any]:
    values = dict(arguments)
    target_date = values.get("date")
    if target_date and (values.get("start_time") or values.get("end_time")):
        raise ValueError("date 不能与 start_time 或 end_time 同时使用")
    if target_date:
        day = target_date if isinstance(target_date, date) else date.fromisoformat(str(target_date))
        values["start_time"] = datetime.combine(day, time.min).astimezone().isoformat()
        values["end_time"] = datetime.combine(day, time.max).astimezone().isoformat()
    start_time = values.get("start_time")
    end_time = values.get("end_time")
    if start_time and end_time:
        start = datetime.fromisoformat(str(start_time).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(end_time).replace("Z", "+00:00"))
        if start.tzinfo is None:
            start = start.astimezone()
        if end.tzinfo is None:
            end = end.astimezone()
        if start > end:
            raise ValueError("start_time 不能晚于 end_time")
        values["start_time"] = start.isoformat()
        values["end_time"] = end.isoformat()
    risk = str(values.get("risk_level") or "").lower()
    if risk and risk not in RISK_LEVELS:
        raise ValueError("risk_level 只能是 none、low、medium 或 high")
    source_type = str(values.get("source_type") or "").lower()
    if source_type and source_type not in SOURCE_TYPES:
        raise ValueError("source_type 只能是 image 或 video")
    review = str(values.get("review_status") or "").lower()
    if review and review not in REVIEW_STATUSES:
        raise ValueError("review_status 值无效")
    values.update(
        risk_level=risk,
        source_type=source_type,
        review_status=review,
    )
    if include_limit:
        limit = int(values.get("limit", 100))
        if not 1 <= limit <= 1000:
            raise ValueError("limit 必须位于 1 到 1000 之间")
        values["limit"] = limit
    return values


def _validate_history_filters(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    return _validate_filters(arguments, include_limit=True)


def _validate_report_filters(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    return _validate_filters(arguments, include_limit=False)


def _validate_review(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    values = dict(arguments)
    action = str(values.get("action") or "").strip().lower()
    if action not in {"confirm", "reject", "close", "reopen"}:
        raise ValueError("action 只能是 confirm、reject、close 或 reopen")
    values["action"] = action
    return values


def _validate_detection_explanation(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    values = dict(arguments)
    question_type = str(values.get("question_type") or "general").strip().lower()
    allowed = {
        "risk_reason",
        "action_advice",
        "similar_history",
        "target_position",
        "general",
    }
    if question_type not in allowed:
        raise ValueError("question_type 必须是风险原因、处置建议、同类历史、目标位置或 general")
    values["question_type"] = question_type
    history_limit = int(values.get("history_limit", 10))
    if not 1 <= history_limit <= 50:
        raise ValueError("history_limit 必须位于 1 到 50 之间")
    values["history_limit"] = history_limit
    return values


def _validate_inspection(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    values = dict(arguments)
    source_type = str(values.get("source_type") or "").lower()
    media_path = values.get("media_path")
    image_path = values.get("image_path")
    video_path = values.get("video_path")
    if image_path and video_path:
        raise ValueError("image_path 与 video_path 只能提供一个")
    if not source_type and media_path:
        suffix = Path(str(media_path)).suffix.lower()
        source_type = "image" if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} else "video"
    if not source_type and image_path:
        source_type = "image"
    if not source_type and video_path:
        source_type = "video"
    if source_type not in SOURCE_TYPES:
        raise ValueError("source_type 必须是 image 或 video")
    path_key = "image_path" if source_type == "image" else "video_path"
    incompatible_key = "video_path" if source_type == "image" else "image_path"
    if values.get(incompatible_key):
        raise ValueError(f"source_type={source_type} 与 {incompatible_key} 冲突")
    values[path_key] = values.get(path_key) or media_path
    if not values.get(path_key):
        raise ValueError(f"缺少 {path_key}")
    values["source_type"] = source_type
    return _validate_image(values) if source_type == "image" else _validate_video(values)


def _validate_probe_video_source(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    values = dict(arguments)
    source_id = str(values.get("source_id") or "").strip().lower()
    if not VIDEO_SOURCE_ID_PATTERN.fullmatch(source_id):
        raise ValueError(
            "source_id 只能包含小写字母、数字、下划线和连字符，且最长 128 个字符"
        )
    values["source_id"] = source_id
    return values


def _validate_capture_video_source(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    values = _validate_probe_video_source(arguments)
    if values.get("duration_seconds") is not None:
        duration = float(values["duration_seconds"])
        if not 1 <= duration <= 3600:
            raise ValueError("duration_seconds 必须在 1 到 3600 之间")
        values["duration_seconds"] = duration
    return values


def _validate_detect_video_source(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    values = _validate_capture_video_source(arguments)
    zone_id = str(values.get("zone_id") or "").strip().lower()
    if zone_id and not VIDEO_SOURCE_ID_PATTERN.fullmatch(zone_id):
        raise ValueError(
            "zone_id 只能包含小写字母、数字、下划线和连字符，且最长 128 个字符"
        )
    parameters = values.get("parameters") or {}
    if zone_id and parameters.get("roi") is not None:
        raise ValueError("zone_id 与 parameters.roi 只能提供一个")
    validated_parameters = _validate_video({"parameters": parameters})["parameters"]
    values["parameters"] = validated_parameters
    if zone_id:
        values["zone_id"] = zone_id
    return values


def _parse_aware_datetime(value: Any, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} 必须是 ISO 8601 时间") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} 必须包含时区")
    return parsed


def _validate_start_monitoring_task(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    values = _validate_detect_video_source(arguments)
    end_time = values.get("end_time")
    run_duration = values.get("run_duration_seconds")
    if bool(end_time) == bool(run_duration is not None):
        raise ValueError("end_time 与 run_duration_seconds 必须且只能提供一个")
    start = (
        _parse_aware_datetime(values["start_time"], "start_time")
        if values.get("start_time")
        else None
    )
    if end_time:
        end = _parse_aware_datetime(end_time, "end_time")
        if start and end <= start:
            raise ValueError("end_time 必须晚于 start_time")
        if start and (end - start).total_seconds() > 86400:
            raise ValueError("非全天候监控任务最长为 24 小时")
    if run_duration is not None:
        duration = float(run_duration)
        if not 1 <= duration <= 86400:
            raise ValueError("run_duration_seconds 必须在 1 到 86400 之间")
        values["run_duration_seconds"] = duration
    capture_duration = values.get("capture_duration_seconds")
    if capture_duration is not None:
        values["capture_duration_seconds"] = float(capture_duration)
    values["interval_seconds"] = float(values.get("interval_seconds", 60.0))
    values["max_consecutive_failures"] = int(
        values.get("max_consecutive_failures", 3)
    )
    return values


def _validate_control_monitoring_task(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    values = dict(arguments)
    action_aliases = {
        "view": "query",
        "show": "query",
        "status": "query",
        "get": "query",
        "cancel": "stop",
    }
    action = str(values.get("action") or "query").strip().lower()
    action = action_aliases.get(action, action)
    if action not in {"query", "stop"}:
        raise ValueError("action 只能是 query 或 stop")
    task_id = str(values.get("task_id") or "").strip().lower()
    if task_id and not re.fullmatch(r"monitor-[a-f0-9]{12}", task_id):
        raise ValueError("task_id 格式无效")
    source_id = str(values.get("source_id") or "").strip().lower()
    if source_id and not VIDEO_SOURCE_ID_PATTERN.fullmatch(source_id):
        raise ValueError("source_id 格式无效")
    values["action"] = action
    if task_id:
        values["task_id"] = task_id
    if source_id:
        values["source_id"] = source_id
    values["limit"] = int(values.get("limit", 10))
    return values


def _validate_start_realtime_inspection(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    values = _validate_detect_video_source(arguments)
    values.pop("duration_seconds", None)
    end_time = values.get("end_time")
    duration = values.get("run_duration_seconds")
    if bool(end_time) == bool(duration is not None):
        raise ValueError("end_time 与 run_duration_seconds 必须且只能提供一个")
    if values.get("start_time"):
        values["start_time"] = _parse_aware_datetime(values["start_time"], "start_time").isoformat(timespec="seconds")
    if end_time:
        end = _parse_aware_datetime(end_time, "end_time")
        if values.get("start_time") and end <= _parse_aware_datetime(values["start_time"], "start_time"):
            raise ValueError("end_time 必须晚于 start_time")
        values["end_time"] = end.isoformat(timespec="seconds")
    else:
        seconds = float(duration)
        if not 1 <= seconds <= 86400: raise ValueError("run_duration_seconds 必须在 1 到 86400 之间")
        values["run_duration_seconds"] = seconds
    parameters = values.get("parameters") or {}
    if "sample_fps" in parameters:
        raise ValueError("实时巡检的 sample_fps 必须使用顶层字段，不能放入 parameters")
    values["sample_fps"] = float(values.get("sample_fps", 2.0))
    if not 0.2 <= values["sample_fps"] <= 10: raise ValueError("sample_fps 必须在 0.2 到 10 之间")
    values["reconnect_interval_seconds"] = float(values.get("reconnect_interval_seconds", 3.0))
    values["max_consecutive_failures"] = int(values.get("max_consecutive_failures", 3))
    values["min_event_hits"] = int(values.get("min_event_hits", 2))
    values["event_silence_seconds"] = float(values.get("event_silence_seconds", 1.0))
    return values


def _validate_control_realtime_inspection(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    values = dict(arguments)
    aliases = {"view": "query", "show": "query", "status": "query", "get": "query", "cancel": "stop"}
    action = aliases.get(str(values.get("action") or "query").strip().lower(), str(values.get("action") or "query").strip().lower())
    if action not in {"query", "stop"}: raise ValueError("action 只能是 query 或 stop")
    task_id = str(values.get("task_id") or "").strip().lower()
    if task_id and not re.fullmatch(r"realtime-[a-f0-9]{12}", task_id): raise ValueError("task_id 格式无效")
    source_id = str(values.get("source_id") or "").strip().lower()
    if source_id and not VIDEO_SOURCE_ID_PATTERN.fullmatch(source_id): raise ValueError("source_id 格式无效")
    for name in ("event_id", "after_event_id"):
        value = str(values.get(name) or "").strip().lower()
        if value and not re.fullmatch(r"realtime-[a-f0-9]{12}-event-\d{4,}", value):
            raise ValueError(f"{name} 格式无效")
        if value: values[name] = value
    values["latest"] = bool(values.get("latest", False))
    values["active_only"] = bool(values.get("active_only", False))
    values["events_only"] = bool(values.get("events_only", False))
    values.update(action=action, limit=int(values.get("limit", 10)))
    if task_id: values["task_id"] = task_id
    if source_id: values["source_id"] = source_id
    return values


def _validate_control_stream_archive(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    values = _validate_probe_video_source(arguments)
    aliases = {
        "view": "query",
        "show": "query",
        "status": "query",
        "get": "query",
        "cancel": "stop",
    }
    action = str(values.get("action") or "query").strip().lower()
    action = aliases.get(action, action)
    if action not in {"start", "stop", "query"}:
        raise ValueError("action 只能是 start、stop 或 query")
    values["action"] = action
    values["segment_seconds"] = float(values.get("segment_seconds", 60.0))
    values["retention_hours"] = float(values.get("retention_hours", 24.0))
    values["limit"] = int(values.get("limit", 100))
    return values


def _validate_detect_archived_video(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    values = _validate_detect_video_source(arguments)
    start = _parse_aware_datetime(values.get("start_time"), "start_time")
    end = _parse_aware_datetime(values.get("end_time"), "end_time")
    if end <= start:
        raise ValueError("end_time 必须晚于 start_time")
    values["start_time"] = start.isoformat(timespec="seconds")
    values["end_time"] = end.isoformat(timespec="seconds")
    values.pop("duration_seconds", None)
    values["coverage_tolerance_seconds"] = float(
        values.get("coverage_tolerance_seconds", 2.0)
    )
    return values


def create_builtin_skill_registry(tools: AgentTools) -> SkillRegistry:
    registry = SkillRegistry()

    registry.register(
        RuntimeSkill(
            SkillSpec(
                "detect-image",
                "检测单张或批量图片中的皮带异物并生成确定性风险结果。",
                required_inputs=("image_path",),
                optional_inputs=("parameters", "line_id", "captured_at", "source_started_at", "source_ended_at"),
                input_schema=ALL_SKILL_SCHEMAS["detect-image"],
            ),
            tools.detect_image,
            _validate_image,
        )
    )
    registry.register(
        RuntimeSkill(
            SkillSpec(
                "detect-video",
                "按抽帧频率、阈值和 ROI 检测视频异物并生成事件与风险结果。",
                required_inputs=("video_path",),
                optional_inputs=("video_start_time", "source_ended_at", "line_id", "start_offset_seconds", "end_offset_seconds", "parameters"),
                input_schema=ALL_SKILL_SCHEMAS["detect-video"],
            ),
            tools.detect_video,
            _validate_video,
        )
    )
    registry.register(
        RuntimeSkill(
            SkillSpec(
                "assess-risk",
                "读取检测 JSON 并返回确定性风险等级、原因和处置建议。",
                optional_inputs=("detection_json", "detection", "source_type"),
                input_schema=ALL_SKILL_SCHEMAS["assess-risk"],
            ),
            tools.assess_risk,
            _validate_risk,
        )
    )
    registry.register(
        RuntimeSkill(
            SkillSpec(
                "parse-detection-result",
                "规范化检测 JSON，并提取目标、事件、位置、置信度、时间和代表帧。",
                optional_inputs=("detection_json", "detection", "source_type"),
                input_schema=ALL_SKILL_SCHEMAS["parse-detection-result"],
            ),
            tools.parse_detection_result,
            _validate_risk,
        )
    )

    def alarm_handler(session_id: str, values: Dict[str, Any]) -> Dict[str, Any]:
        action = values.pop("action")
        if action == "confirm":
            return tools.confirm_alarm(session_id, values)
        if action == "cancel":
            return tools.cancel_alarm(session_id, values)
        return tools.current_alarm(session_id, values)

    registry.register(
        RuntimeSkill(
            SkillSpec(
                "control-alarm",
                (
                    "查看、确认或取消当前报警，并记录操作审计；也可在用户明确要求时按"
                    "实时巡检 task_id 批量闭环本轮报警。查询必须使用 query；"
                    "confirm/cancel 仅用于明确写操作。"
                ),
                optional_inputs=("action", "alarm_id", "scope", "task_id", "line_id", "session_only", "note"),
                safety="controlled-write",
                input_schema=ALL_SKILL_SCHEMAS["control-alarm"],
            ),
            alarm_handler,
            _validate_alarm,
        )
    )
    registry.register(
        RuntimeSkill(
            SkillSpec(
                "query-history",
                "按时间、风险等级、线路、来源和复核状态查询检测历史。",
                optional_inputs=("date", "start_time", "end_time", "risk_level", "line_id", "source_type", "review_status", "limit", "include_details"),
                input_schema=ALL_SKILL_SCHEMAS["query-history"],
            ),
            tools.query_history,
            _validate_history_filters,
        )
    )
    registry.register(
        RuntimeSkill(
            SkillSpec(
                "generate-risk-report",
                "按日期或筛选条件汇总检测、风险、报警、异物和闭环状态。",
                optional_inputs=("date", "start_time", "end_time", "risk_level", "line_id", "source_type", "review_status"),
                input_schema=ALL_SKILL_SCHEMAS["generate-risk-report"],
            ),
            tools.generate_risk_report,
            _validate_report_filters,
        )
    )
    registry.register(
        RuntimeSkill(
            SkillSpec(
                "review-detection",
                "人工确认、驳回、关闭或重新打开检测结果并保留审计记录。",
                required_inputs=("action",),
                optional_inputs=("detection_id", "reviewer", "note"),
                safety="controlled-write",
                input_schema=ALL_SKILL_SCHEMAS["review-detection"],
            ),
            tools.review_detection,
            _validate_review,
        )
    )
    registry.register(
        RuntimeSkill(
            SkillSpec(
                "explain-detection-result",
                (
                    "基于当前会话指定或最近一次 detection_id 的数据库事实，解释风险原因、"
                    "处置建议、同类历史或目标位置。只读；风险等级和报警状态仍以规则引擎与数据库为准。"
                ),
                optional_inputs=(
                    "detection_id",
                    "question",
                    "question_type",
                    "history_limit",
                ),
                input_schema=ALL_SKILL_SCHEMAS["explain-detection-result"],
            ),
            tools.explain_detection_result,
            _validate_detection_explanation,
        )
    )

    def inspection_handler(session_id: str, values: Dict[str, Any]) -> Dict[str, Any]:
        source_type = values.pop("source_type")
        values.pop("media_path", None)
        result = (
            tools.detect_image(session_id, values)
            if source_type == "image"
            else tools.detect_video(session_id, values)
        )
        result.setdefault("data", {})["workflow"] = [
            f"detect-{source_type}",
            "assess-risk",
            "persist-history",
            "create-alarm",
        ]
        return result

    registry.register(
        RuntimeSkill(
            SkillSpec(
                "run-inspection-task",
                "编排图片或视频检测、风险研判、历史入库和报警创建。",
                optional_inputs=(
                    "media_path", "source_type", "image_path", "video_path",
                    "video_start_time", "source_ended_at", "captured_at",
                    "source_started_at", "line_id", "start_offset_seconds", "end_offset_seconds", "parameters",
                ),
                input_schema=ALL_SKILL_SCHEMAS["run-inspection-task"],
            ),
            inspection_handler,
            _validate_inspection,
        )
    )
    registry.register(
        RuntimeSkill(
            SkillSpec(
                "probe-video-source",
                (
                    "检查已注册 RTSP 视频源是否在线，并读取一帧返回连接延迟、"
                    "分辨率、FPS、编码格式和安全错误码；不执行异物检测。"
                ),
                required_inputs=("source_id",),
                input_schema=ALL_SKILL_SCHEMAS["probe-video-source"],
            ),
            tools.probe_video_source,
            _validate_probe_video_source,
        )
    )
    registry.register(
        RuntimeSkill(
            SkillSpec(
                "capture-video-source",
                (
                    "从已注册 RTSP 视频源采集一个定长本地 MP4 片段并返回真实起止时间、"
                    "帧数和安全元数据；不执行异物检测。"
                ),
                required_inputs=("source_id",),
                optional_inputs=("duration_seconds",),
                safety="local-write",
                input_schema=ALL_SKILL_SCHEMAS["capture-video-source"],
            ),
            tools.capture_video_source,
            _validate_capture_video_source,
        )
    )
    registry.register(
        RuntimeSkill(
            SkillSpec(
                "detect-video-source",
                (
                    "按需采集已注册 RTSP 视频源并调用现有视频检测、风险研判、报警报告"
                    "和历史入库；可通过 zone_id 使用已注册 ROI。"
                ),
                required_inputs=("source_id",),
                optional_inputs=("duration_seconds", "zone_id", "parameters"),
                safety="local-write",
                input_schema=ALL_SKILL_SCHEMAS["detect-video-source"],
            ),
            tools.detect_video_source,
            _validate_detect_video_source,
        )
    )
    registry.register(
        RuntimeSkill(
            SkillSpec(
                "start-monitoring-task",
                (
                    "为已注册 RTSP 源创建有明确结束条件的非全天候后台监控任务，"
                    "按轮执行实时检测并在连续失败达到上限时自动停止。"
                ),
                required_inputs=("source_id",),
                optional_inputs=(
                    "start_time",
                    "end_time",
                    "run_duration_seconds",
                    "capture_duration_seconds",
                    "interval_seconds",
                    "zone_id",
                    "parameters",
                    "max_consecutive_failures",
                ),
                safety="controlled-write",
                input_schema=ALL_SKILL_SCHEMAS["start-monitoring-task"],
            ),
            tools.start_monitoring_task,
            _validate_start_monitoring_task,
        )
    )
    registry.register(
        RuntimeSkill(
            SkillSpec(
                "control-monitoring-task",
                "查看或停止当前会话的非全天候监控任务；查看必须使用 query。",
                optional_inputs=("action", "task_id", "source_id", "limit"),
                safety="controlled-write",
                input_schema=ALL_SKILL_SCHEMAS["control-monitoring-task"],
            ),
            tools.control_monitoring_task,
            _validate_control_monitoring_task,
        )
    )
    registry.register(
        RuntimeSkill(
            SkillSpec(
                "start-realtime-inspection",
                "以单一持续 RTSP 连接执行有界实时巡检、抽帧单帧推理、事件聚合、确定性风险研判、报警和历史入库；不生成正常画面 MP4。",
                required_inputs=("source_id",),
                optional_inputs=("start_time", "end_time", "run_duration_seconds", "sample_fps", "zone_id", "parameters",
                                 "reconnect_interval_seconds", "max_consecutive_failures", "min_event_hits", "event_silence_seconds"),
                safety="controlled-write",
                input_schema=ALL_SKILL_SCHEMAS["start-realtime-inspection"],
            ),
            tools.start_realtime_inspection,
            _validate_start_realtime_inspection,
        )
    )
    registry.register(
        RuntimeSkill(
            SkillSpec(
                "control-realtime-inspection",
                "查询或停止当前会话的持续实时巡检；查看、显示、状态统一使用 query，只有明确停止请求才使用 stop。",
                optional_inputs=("action", "task_id", "source_id", "event_id", "after_event_id",
                                 "latest", "active_only", "events_only", "task_only",
                                 "compact", "limit"),
                safety="controlled-write",
                input_schema=ALL_SKILL_SCHEMAS["control-realtime-inspection"],
            ),
            tools.control_realtime_inspection,
            _validate_control_realtime_inspection,
        )
    )
    registry.register(
        RuntimeSkill(
            SkillSpec(
                "control-stream-archive",
                (
                    "启动、停止或查询已注册 RTSP 视频源的持续历史录像归档。"
                    "查看状态必须使用 query；持续归档只采集，不运行 YOLO。"
                ),
                required_inputs=("source_id",),
                optional_inputs=(
                    "action",
                    "segment_seconds",
                    "retention_hours",
                    "limit",
                ),
                safety="controlled-write",
                input_schema=ALL_SKILL_SCHEMAS["control-stream-archive"],
            ),
            tools.control_stream_archive,
            _validate_control_stream_archive,
        )
    )
    registry.register(
        RuntimeSkill(
            SkillSpec(
                "detect-archived-video",
                (
                    "按带时区的绝对时间范围查找已归档监控录像，验证覆盖完整性，"
                    "裁剪边界片段并复用现有视频检测、风险研判、报警和历史入库。"
                    "录像有缺口或文件缺失时必须拒绝检测，不能用当前实时画面替代。"
                ),
                required_inputs=("source_id", "start_time", "end_time"),
                optional_inputs=(
                    "zone_id",
                    "parameters",
                    "coverage_tolerance_seconds",
                ),
                safety="local-write",
                input_schema=ALL_SKILL_SCHEMAS["detect-archived-video"],
            ),
            tools.detect_archived_video,
            _validate_detect_archived_video,
        )
    )
    return registry
