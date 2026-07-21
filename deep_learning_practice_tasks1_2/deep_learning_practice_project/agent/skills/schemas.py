from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping


def _object(properties: Mapping[str, Any], *, required: tuple[str, ...] = ()) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


def _string(description: str, *, max_length: int = 2048) -> Dict[str, Any]:
    return {
        "type": "string",
        "minLength": 1,
        "maxLength": max_length,
        "description": description,
    }


def _datetime(description: str) -> Dict[str, Any]:
    return {**_string(description, max_length=64), "format": "date-time"}


DATE_SCHEMA = {
    **_string("本地日历日期，格式 YYYY-MM-DD；相对日期必须在规划阶段转换。", max_length=10),
    "format": "date",
}
LINE_ID_SCHEMA = _string("规范线路标识；必须来自线路注册表或已有记录，不能由模型猜测。", max_length=128)
RISK_LEVEL_SCHEMA = {
    "type": "string",
    "enum": ["none", "low", "medium", "high"],
    "description": "风险等级过滤条件，必须输出英文规范值。",
    "aliases": {
        "无风险": "none",
        "无": "none",
        "低风险": "low",
        "低": "low",
        "中风险": "medium",
        "中": "medium",
        "高风险": "high",
        "高": "high",
    },
}
SOURCE_TYPE_SCHEMA = {
    "type": "string",
    "enum": ["image", "video"],
    "description": "媒体来源类型，必须输出 image 或 video。",
    "aliases": {"图片": "image", "图像": "image", "照片": "image", "视频": "video"},
}
AUTO_SOURCE_TYPE_SCHEMA = {
    **deepcopy(SOURCE_TYPE_SCHEMA),
    "enum": ["auto", "image", "video"],
    "default": "auto",
    "description": "检测 JSON 来源类型；无法确定时使用 auto。",
    "aliases": {**SOURCE_TYPE_SCHEMA["aliases"], "自动": "auto"},
}
REVIEW_STATUS_SCHEMA = {
    "type": "string",
    "enum": ["unreviewed", "confirmed", "rejected", "closed"],
    "description": "人工复核状态过滤条件。",
    "aliases": {
        "待复核": "unreviewed",
        "未复核": "unreviewed",
        "已确认": "confirmed",
        "已驳回": "rejected",
        "误报": "rejected",
        "已闭环": "closed",
        "已关闭": "closed",
    },
}


def _ratio(description: str, default: float) -> Dict[str, Any]:
    return {
        "type": "number",
        "exclusiveMinimum": 0,
        "exclusiveMaximum": 1,
        "default": default,
        "description": description,
    }


COMMON_DETECTION_PARAMETER_PROPERTIES = {
    "conf": {
        "type": "number",
        "minimum": 0,
        "exclusiveMaximum": 1,
        "default": 0.25,
        "description": "最低候选置信度；必须小于 known_conf。",
    },
    "known_conf": {
        "type": "number",
        "exclusiveMinimum": 0,
        "maximum": 1,
        "default": 0.40,
        "description": "确认已知类别的最低置信度；必须大于 conf。",
    },
    "imgsz": {
        "type": "integer",
        "minimum": 32,
        "maximum": 4096,
        "default": 800,
        "description": "YOLO 推理尺寸。",
    },
    "nms_iou": _ratio("NMS IoU 阈值。", 0.40),
    "duplicate_iou": _ratio("同类重复框 IoU 阈值。", 0.45),
    "duplicate_containment": _ratio("同类包含关系去重阈值。", 0.80),
}

IMAGE_PARAMETER_PROPERTIES = {
    **deepcopy(COMMON_DETECTION_PARAMETER_PROPERTIES),
    "cross_class_iou": _ratio("跨类别重叠仲裁 IoU 阈值。", 0.70),
    "cross_class_containment": _ratio("跨类别包含关系仲裁阈值。", 0.92),
    "max_area_ratio": _ratio("相对整图的最大允许检测框面积比例。", 0.65),
    "confirm_low_confidence_unknown": {
        "type": "boolean",
        "default": False,
        "description": "是否把低置信度结果直接确认为未知异物；默认仅保留为待确认候选。",
    },
}
IMAGE_PARAMETERS_SCHEMA = _object(IMAGE_PARAMETER_PROPERTIES)

VIDEO_PARAMETER_PROPERTIES = {
    **deepcopy(COMMON_DETECTION_PARAMETER_PROPERTIES),
    "sample_fps": {
        "type": "number",
        "exclusiveMinimum": 0,
        "maximum": 60,
        "default": 4.0,
        "description": "每秒抽取并检测的帧数；不能超过 60。",
    },
    "agnostic_nms": {
        "type": "boolean",
        "default": False,
        "description": "是否启用跨类别 NMS。",
    },
    "event_silence_seconds": {
        "type": "number",
        "exclusiveMinimum": 0,
        "default": 1.0,
        "description": "连续事件之间允许的最大静默时间。",
    },
    "track_max_age_seconds": {
        "type": "number",
        "minimum": 0,
        "default": 1.0,
        "description": "目标轨迹允许失联的最大秒数。",
    },
    "min_unknown_hits": {
        "type": "integer",
        "minimum": 1,
        "default": 2,
        "description": "低置信度未知目标被确认前所需的连续命中次数。",
    },
    "unknown_single_frame_conf": {
        "type": "number",
        "minimum": 0,
        "maximum": 1,
        "default": 0.40,
        "description": "未知目标单帧确认阈值；必须位于 conf 与 known_conf 之间。",
    },
    "track_iou": _ratio("轨迹关联的最低 IoU。", 0.15),
    "track_center_distance_ratio": {
        "type": "number",
        "exclusiveMinimum": 0,
        "default": 3.0,
        "description": "轨迹关联允许的中心距离与框尺寸比值。",
    },
    "roi": {
        "type": ["array", "null"],
        "items": {"type": "integer", "minimum": 0},
        "minItems": 4,
        "maxItems": 4,
        "default": None,
        "description": "全帧像素坐标 [x1, y1, x2, y2]；x2>x1 且 y2>y1。",
    },
}
VIDEO_PARAMETERS_SCHEMA = _object(VIDEO_PARAMETER_PROPERTIES)

DETECT_IMAGE_SCHEMA = _object(
    {
        "image_path": _string("待检测图片的现有本地路径。"),
        "parameters": {**deepcopy(IMAGE_PARAMETERS_SCHEMA), "description": "图片检测参数。"},
        "line_id": deepcopy(LINE_ID_SCHEMA),
        "captured_at": _datetime("图片拍摄时间。"),
        "source_started_at": _datetime("媒体来源开始时间。"),
        "source_ended_at": _datetime("媒体来源结束时间。"),
    },
    required=("image_path",),
)

DETECT_VIDEO_SCHEMA = _object(
    {
        "video_path": _string("待检测视频的现有本地路径。"),
        "video_start_time": _datetime("原始视频第 0 秒对应的真实时间。"),
        "source_ended_at": _datetime("已知的原始视频结束时间。"),
        "line_id": deepcopy(LINE_ID_SCHEMA),
        "start_offset_seconds": {
            "type": "number",
            "minimum": 0,
            "default": 0.0,
            "description": "从原视频起点开始计算的检测片段开始秒数。",
        },
        "end_offset_seconds": {
            "type": "number",
            "exclusiveMinimum": 0,
            "description": "检测片段结束秒数；必须大于 start_offset_seconds，省略表示直到视频结束。",
        },
        "parameters": {**deepcopy(VIDEO_PARAMETERS_SCHEMA), "description": "视频检测参数。"},
    },
    required=("video_path",),
)

DETECTION_JSON_INPUT_SCHEMA = _object(
    {
        "detection_json": _string("已有检测结果 JSON 文件路径。"),
        "detection": {"type": "object", "description": "内存中的检测结果对象。"},
        "source_type": deepcopy(AUTO_SOURCE_TYPE_SCHEMA),
    }
)
DETECTION_JSON_INPUT_SCHEMA["anyOf"] = [
    {"required": ["detection_json"]},
    {"required": ["detection"]},
]

CONTROL_ALARM_SCHEMA = _object(
    {
        "action": {
            "type": "string",
            "enum": ["query", "confirm", "cancel"],
            "default": "query",
            "description": (
                "报警操作。查看、查询、显示、获取状态必须使用 query；"
                "confirm/cancel 仅用于用户明确要求的写操作。"
            ),
            "aliases": {"view": "query", "show": "query", "get": "query", "status": "query"},
        },
        "alarm_id": _string("可选的明确报警 ID。", max_length=128),
        "line_id": deepcopy(LINE_ID_SCHEMA),
        "session_only": {
            "type": "boolean",
            "default": False,
            "description": "为 true 时仅查询当前会话报警。",
        },
        "note": _string("确认或取消报警时写入审计记录的说明。", max_length=1000),
    }
)

HISTORY_FILTER_PROPERTIES = {
    "date": deepcopy(DATE_SCHEMA),
    "start_time": _datetime("查询范围开始时间，必须带时区。"),
    "end_time": _datetime("查询范围结束时间，必须带时区。"),
    "risk_level": deepcopy(RISK_LEVEL_SCHEMA),
    "line_id": deepcopy(LINE_ID_SCHEMA),
    "source_type": deepcopy(SOURCE_TYPE_SCHEMA),
    "review_status": deepcopy(REVIEW_STATUS_SCHEMA),
}
QUERY_HISTORY_SCHEMA = _object(
    {
        **deepcopy(HISTORY_FILTER_PROPERTIES),
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": 1000,
            "default": 100,
            "description": "最多返回的记录数量。",
        },
        "include_details": {
            "type": "boolean",
            "default": True,
            "description": "是否返回检测 JSON、报警报告和代表帧等详细证据。",
        },
    }
)
GENERATE_RISK_REPORT_SCHEMA = _object(deepcopy(HISTORY_FILTER_PROPERTIES))

REVIEW_DETECTION_SCHEMA = _object(
    {
        "detection_id": _string("待复核的检测记录 ID；省略时仅可指代当前会话最新记录。", max_length=128),
        "action": {
            "type": "string",
            "enum": ["confirm", "reject", "close", "reopen"],
            "description": "明确的人工复核动作；不得从模糊表达推断。",
        },
        "reviewer": _string("执行人工复核的人员标识。", max_length=128),
        "note": _string("人工复核或闭环处置说明。", max_length=1000),
    },
    required=("action",),
)

_inspection_parameter_properties = {
    name: {key: deepcopy(value) for key, value in schema.items() if key != "default"}
    for name, schema in {**IMAGE_PARAMETER_PROPERTIES, **VIDEO_PARAMETER_PROPERTIES}.items()
}
INSPECTION_PARAMETERS_SCHEMA = _object(_inspection_parameter_properties)
RUN_INSPECTION_SCHEMA = _object(
    {
        "media_path": _string("待检测图片或视频的统一媒体路径。"),
        "source_type": deepcopy(SOURCE_TYPE_SCHEMA),
        "image_path": _string("待检测图片路径。"),
        "video_path": _string("待检测视频路径。"),
        "video_start_time": _datetime("原始视频第 0 秒对应的真实时间。"),
        "source_ended_at": _datetime("媒体来源结束时间。"),
        "captured_at": _datetime("图片拍摄时间。"),
        "source_started_at": _datetime("媒体来源开始时间。"),
        "line_id": deepcopy(LINE_ID_SCHEMA),
        "start_offset_seconds": deepcopy(DETECT_VIDEO_SCHEMA["properties"]["start_offset_seconds"]),
        "end_offset_seconds": deepcopy(DETECT_VIDEO_SCHEMA["properties"]["end_offset_seconds"]),
        "parameters": {
            **INSPECTION_PARAMETERS_SCHEMA,
            "description": "参数必须与实际 source_type 对应；执行层会进行二次白名单校验。",
        },
    }
)
RUN_INSPECTION_SCHEMA["anyOf"] = [
    {"required": ["media_path"]},
    {"required": ["image_path"]},
    {"required": ["video_path"]},
]

PROBE_VIDEO_SOURCE_SCHEMA = _object(
    {
        "source_id": {
            **_string(
                "已注册的视频源规范标识，例如 main-monitor；不得传入 RTSP URL。",
                max_length=128,
            ),
            "pattern": "^[a-z0-9][a-z0-9_-]{0,127}$",
        },
    },
    required=("source_id",),
)

CAPTURE_VIDEO_SOURCE_SCHEMA = _object(
    {
        "source_id": deepcopy(PROBE_VIDEO_SOURCE_SCHEMA["properties"]["source_id"]),
        "duration_seconds": {
            "type": "number",
            "minimum": 1,
            "maximum": 3600,
            "description": (
                "本次实时采集时长（秒）；省略时使用视频源配置中的 capture_window_seconds。"
            ),
        },
    },
    required=("source_id",),
)

DETECT_VIDEO_SOURCE_SCHEMA = _object(
    {
        "source_id": deepcopy(PROBE_VIDEO_SOURCE_SCHEMA["properties"]["source_id"]),
        "duration_seconds": deepcopy(
            CAPTURE_VIDEO_SOURCE_SCHEMA["properties"]["duration_seconds"]
        ),
        "zone_id": {
            **_string(
                "视频源注册表中的区域标识；执行层将其确定性转换为 ROI。",
                max_length=128,
            ),
            "pattern": "^[a-z0-9][a-z0-9_-]{0,127}$",
        },
        "parameters": {
            **deepcopy(VIDEO_PARAMETERS_SCHEMA),
            "description": "交给现有视频检测链路的严格参数。",
        },
    },
    required=("source_id",),
)


ALL_SKILL_SCHEMAS = {
    "detect-image": DETECT_IMAGE_SCHEMA,
    "detect-video": DETECT_VIDEO_SCHEMA,
    "assess-risk": DETECTION_JSON_INPUT_SCHEMA,
    "parse-detection-result": DETECTION_JSON_INPUT_SCHEMA,
    "control-alarm": CONTROL_ALARM_SCHEMA,
    "query-history": QUERY_HISTORY_SCHEMA,
    "generate-risk-report": GENERATE_RISK_REPORT_SCHEMA,
    "review-detection": REVIEW_DETECTION_SCHEMA,
    "run-inspection-task": RUN_INSPECTION_SCHEMA,
    "probe-video-source": PROBE_VIDEO_SOURCE_SCHEMA,
    "capture-video-source": CAPTURE_VIDEO_SOURCE_SCHEMA,
    "detect-video-source": DETECT_VIDEO_SOURCE_SCHEMA,
}
