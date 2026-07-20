from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from agent.tools import AgentTools

from .base import RuntimeSkill, SkillRegistry, SkillSpec


RISK_LEVELS = {"none", "low", "medium", "high"}
SOURCE_TYPES = {"image", "video"}
REVIEW_STATUSES = {"unreviewed", "confirmed", "rejected", "closed"}
ALARM_ACTIONS = {"query", "confirm", "cancel"}
ALARM_READ_ACTION_ALIASES = {
    "view": "query",
    "show": "query",
    "get": "query",
    "status": "query",
}
CONTROL_ALARM_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["query", "confirm", "cancel"],
            "default": "query",
            "description": (
                "报警操作。查看、查询、显示、获取当前报警或报警状态时必须传 query；"
                "只有用户明确要求确认或取消时才能分别传 confirm 或 cancel。"
            ),
            "aliases": dict(ALARM_READ_ACTION_ALIASES),
        },
        "alarm_id": {
            "type": "string",
            "description": "可选的明确报警 ID；未提供时查询或操作当前报警。",
        },
        "line_id": {
            "type": "string",
            "description": "查询当前报警时使用的可选线路标识。",
        },
        "session_only": {
            "type": "boolean",
            "default": False,
            "description": "为 true 时仅查询当前会话的报警。",
        },
        "note": {
            "type": "string",
            "description": "确认或取消报警时写入审计记录的可选操作说明。",
        },
    },
    "required": [],
    "additionalProperties": False,
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
        "track_center_distance_ratio",
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
    if "sample_fps" in parameters and float(parameters["sample_fps"]) <= 0:
        raise ValueError("sample_fps 必须大于 0")
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
    values["action"] = action
    return values


def _validate_filters(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    values = dict(arguments)
    risk = str(values.get("risk_level") or "").lower()
    if risk and risk not in RISK_LEVELS:
        raise ValueError("risk_level 只能是 none、low、medium 或 high")
    source_type = str(values.get("source_type") or "").lower()
    if source_type and source_type not in SOURCE_TYPES:
        raise ValueError("source_type 只能是 image 或 video")
    review = str(values.get("review_status") or "").lower()
    if review and review not in REVIEW_STATUSES:
        raise ValueError("review_status 值无效")
    limit = int(values.get("limit", 100))
    if not 1 <= limit <= 1000:
        raise ValueError("limit 必须位于 1 到 1000 之间")
    values.update(
        risk_level=risk,
        source_type=source_type,
        review_status=review,
        limit=limit,
    )
    return values


def _validate_review(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    values = dict(arguments)
    action = str(values.get("action") or "confirm").lower()
    if action not in {"confirm", "reject", "close", "reopen"}:
        raise ValueError("action 只能是 confirm、reject、close 或 reopen")
    values["action"] = action
    return values


def _validate_inspection(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    values = dict(arguments)
    source_type = str(values.get("source_type") or "").lower()
    media_path = values.get("media_path")
    if not source_type and media_path:
        suffix = Path(str(media_path)).suffix.lower()
        source_type = "image" if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} else "video"
    if source_type not in SOURCE_TYPES:
        raise ValueError("source_type 必须是 image 或 video")
    path_key = "image_path" if source_type == "image" else "video_path"
    values[path_key] = values.get(path_key) or media_path
    if not values.get(path_key):
        raise ValueError(f"缺少 {path_key}")
    values["source_type"] = source_type
    return _validate_image(values) if source_type == "image" else _validate_video(values)


def create_builtin_skill_registry(tools: AgentTools) -> SkillRegistry:
    registry = SkillRegistry()

    registry.register(
        RuntimeSkill(
            SkillSpec(
                "detect-image",
                "检测单张或批量图片中的皮带异物并生成确定性风险结果。",
                required_inputs=("image_path",),
                optional_inputs=("parameters", "line_id", "captured_at", "source_started_at", "source_ended_at"),
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
                optional_inputs=("video_start_time", "source_ended_at", "line_id", "parameters"),
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
                    "查看、确认或取消当前报警，并记录操作审计。查看、查询、显示或获取"
                    "报警状态时 action 必须为 query；confirm/cancel 仅用于用户明确要求的写操作。"
                ),
                optional_inputs=("action", "alarm_id", "line_id", "session_only", "note"),
                safety="controlled-write",
                input_schema=CONTROL_ALARM_INPUT_SCHEMA,
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
                optional_inputs=("start_time", "end_time", "risk_level", "line_id", "source_type", "review_status", "limit", "include_details"),
            ),
            tools.query_history,
            _validate_filters,
        )
    )
    registry.register(
        RuntimeSkill(
            SkillSpec(
                "generate-risk-report",
                "按日期或筛选条件汇总检测、风险、报警、异物和闭环状态。",
                optional_inputs=("date", "start_time", "end_time", "risk_level", "line_id", "source_type", "review_status", "limit"),
            ),
            tools.generate_risk_report,
            _validate_filters,
        )
    )
    registry.register(
        RuntimeSkill(
            SkillSpec(
                "review-detection",
                "人工确认、驳回、关闭或重新打开检测结果并保留审计记录。",
                optional_inputs=("detection_id", "action", "reviewer", "note"),
                safety="controlled-write",
            ),
            tools.review_detection,
            _validate_review,
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
                    "source_started_at", "line_id", "parameters",
                ),
            ),
            inspection_handler,
            _validate_inspection,
        )
    )
    return registry
