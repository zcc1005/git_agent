from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SCHEMA_VERSION = "1.0"
CONVERTER_NAME = "task3_alarm.unified_alarm/1.0"
DEFAULT_SCHEMA_PATH = Path(__file__).with_name("unified_alarm_schema.json")


class UnifiedAlarmValidationError(ValueError):
    """Raised when a converted alarm document violates required invariants."""


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到检测结果 JSON：{path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"检测结果 JSON 解析失败：{path}，第 {exc.lineno} 行：{exc.msg}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"检测结果必须是 JSON 对象：{path}")
    return data


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_dict_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _as_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return list(
        dict.fromkeys(str(item) for item in value if item not in (None, ""))
    )


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clean_count_map(value: Any) -> Dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: Dict[str, int] = {}
    for key, raw_count in value.items():
        result[str(key)] = max(0, _int(raw_count))
    return result


def _bbox(raw_object: Dict[str, Any]) -> List[float]:
    value = raw_object.get("bbox_xyxy") or raw_object.get("bbox") or []
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    return [round(_float(item), 2) for item in value[:4]]


def _bbox_area(bbox: Sequence[float]) -> float:
    if len(bbox) < 4:
        return 0.0
    return round(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]), 2)


def _resolve_media_path(path_value: str, input_json: Path) -> Optional[Path]:
    if not path_value:
        return None
    raw_path = Path(path_value)
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates = [PROJECT_ROOT / raw_path, input_json.parent / raw_path, Path.cwd() / raw_path]
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate.resolve()
        except OSError:
            continue
    return None


def _image_size(path_value: str, input_json: Path) -> Optional[Tuple[float, float]]:
    image_path = _resolve_media_path(path_value, input_json)
    if image_path is None:
        return None
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size
        return float(width), float(height)
    except Exception:
        return None


def _detection_extent(objects: Iterable[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    max_x = 0.0
    max_y = 0.0
    for raw_object in objects:
        bbox = _bbox(raw_object)
        max_x = max(max_x, bbox[2])
        max_y = max(max_y, bbox[3])
    if max_x <= 0 or max_y <= 0:
        return None
    return max_x, max_y


def _estimate_position(bbox: Sequence[float], width: float, height: float) -> str:
    if len(bbox) < 4 or width <= 0 or height <= 0:
        return "未知区域"
    x_center = (bbox[0] + bbox[2]) / 2.0
    y_center = (bbox[1] + bbox[3]) / 2.0
    if x_center < width / 3.0:
        x_part = "左侧"
    elif x_center > width * 2.0 / 3.0:
        x_part = "右侧"
    else:
        x_part = "中间"
    if y_center < height / 3.0:
        y_part = "上部"
    elif y_center > height * 2.0 / 3.0:
        y_part = "下部"
    else:
        y_part = "中部"
    return f"{y_part}{x_part}区域"


def _normalize_object(
    raw_object: Dict[str, Any],
    object_id: str,
    source_frame: str,
    input_json: Path,
    frame_objects: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    bbox = _bbox(raw_object)
    raw_position = str(raw_object.get("position") or "").strip()
    if raw_position:
        position = raw_position
        position_basis = "input"
    else:
        dimensions = _image_size(source_frame, input_json)
        if dimensions is not None:
            position = _estimate_position(bbox, *dimensions)
            position_basis = "image_dimensions"
        else:
            dimensions = _detection_extent(frame_objects)
            if dimensions is not None:
                position = _estimate_position(bbox, *dimensions)
                position_basis = "detection_extent"
            else:
                position = "未知区域"
                position_basis = "unavailable"

    raw_class = str(raw_object.get("class") or "unknown")
    raw_class_name = str(raw_object.get("class_name") or raw_class or "未知异物")
    raw_class_id = raw_object.get("class_id")
    class_id = _int(raw_class_id) if raw_class_id not in (None, "") else None
    supplied_area = _optional_float(raw_object.get("area"))
    area = supplied_area if supplied_area is not None and supplied_area >= 0 else _bbox_area(bbox)

    return {
        "object_id": object_id,
        "track_id": raw_object.get("track_id"),
        "source_frame": source_frame,
        "class_id": class_id,
        "class": raw_class,
        "class_name": raw_class_name,
        "confidence": round(min(1.0, max(0.0, _float(raw_object.get("confidence")))), 4),
        "bbox_xyxy": bbox,
        "area": round(area, 2),
        "position": position,
        "position_basis": position_basis,
        "rejected_as_unknown": bool(raw_object.get("rejected_as_unknown", raw_class == "unknown")),
        "predicted_class": (
            str(raw_object["predicted_class"]) if raw_object.get("predicted_class") is not None else None
        ),
        "predicted_class_name": (
            str(raw_object["predicted_class_name"])
            if raw_object.get("predicted_class_name") is not None
            else None
        ),
    }


def _pending_risk() -> Dict[str, Any]:
    return {
        "status": "pending",
        "level": None,
        "code": None,
        "reason": None,
        "requires_stop": None,
        "action_code": None,
    }


def _pending_report() -> Dict[str, Any]:
    return {
        "status": "pending",
        "conclusion": None,
        "risk_explanation": None,
        "recommended_action": None,
        "generator": None,
    }


def _safe_report_id(source_type: str, source_path: str, created_at: str) -> str:
    stem = Path(source_path).stem or source_type
    raw = f"alarm-{source_type}-{created_at}-{stem}"
    cleaned = re.sub(r"[^0-9A-Za-z_-]+", "-", raw).strip("-")
    return cleaned[:160] or f"alarm-{source_type}"


def _object_class_counts(objects: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter(str(item.get("class_name") or item.get("class") or "未知异物") for item in objects)
    return dict(counts)


def _event_summary(
    objects: Sequence[Dict[str, Any]],
    reported_peak_box_count: int,
    positive_sample_count: int,
    unique_object_count: Optional[int] = None,
    track_ids: Optional[Sequence[Any]] = None,
    class_counts: Optional[Dict[str, int]] = None,
    observed_classes: Optional[Sequence[str]] = None,
    max_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    if max_confidence is None and objects:
        max_confidence = max(_float(item.get("confidence")) for item in objects)
    effective_class_counts = class_counts or _object_class_counts(objects)
    effective_observed = list(dict.fromkeys(observed_classes or effective_class_counts.keys()))
    effective_track_ids = list(
        dict.fromkeys(
            item
            for item in (track_ids or [obj.get("track_id") for obj in objects])
            if item is not None
        )
    )
    if unique_object_count is None:
        unique_object_count = len(effective_track_ids) or len(objects)
    return {
        "detection_box_count": len(objects),
        "reported_peak_box_count": max(0, reported_peak_box_count),
        "unique_object_count": max(0, unique_object_count),
        "positive_sample_count": max(0, positive_sample_count),
        "track_ids": effective_track_ids,
        "class_counts": effective_class_counts,
        "observed_classes": effective_observed,
        "max_confidence": round(max_confidence, 4) if max_confidence is not None else None,
    }


def _base_document(
    source_type: str,
    source_path: str,
    created_at: str,
    start_real_time: Optional[str],
    end_real_time: Optional[str],
    duration_seconds: Optional[float],
    status: str,
    events: List[Dict[str, Any]],
    detection_box_count: int,
    positive_frame_count: int,
    class_counts: Dict[str, int],
    input_format: str,
    input_json: Path,
    model_path: str,
    thresholds: Dict[str, Any],
    warnings: List[str],
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "report_id": _safe_report_id(source_type, source_path, created_at),
        "created_at": created_at,
        "source": {
            "type": source_type,
            "name": Path(source_path).name,
            "path": source_path,
            "start_real_time": start_real_time,
            "end_real_time": end_real_time,
            "duration_seconds": duration_seconds,
        },
        "detection_summary": {
            "status": status,
            "has_foreign_object": bool(events),
            "event_count": len(events),
            "detection_box_count": max(0, detection_box_count),
            "positive_frame_count": max(0, positive_frame_count),
            "class_counts": class_counts,
        },
        "events": events,
        "overall_risk": _pending_risk(),
        "generated_report": _pending_report(),
        "provenance": {
            "input_format": input_format,
            "input_json": str(input_json.resolve()),
            "model_path": model_path,
            "thresholds": thresholds,
            "converter": CONVERTER_NAME,
            "conversion_warnings": warnings,
        },
    }


def convert_image_detection(detection: Dict[str, Any], input_json: Path) -> Dict[str, Any]:
    raw_objects = _as_dict_list(detection.get("objects"))
    source_path = str(detection.get("source") or "")
    created_at = str(detection.get("timestamp") or detection.get("created_at") or datetime.now().isoformat(timespec="seconds"))
    warnings: List[str] = []

    grouped: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for raw_object in raw_objects:
        image_path = str(raw_object.get("image") or raw_object.get("image_name") or source_path)
        grouped.setdefault(image_path, []).append(raw_object)

    events: List[Dict[str, Any]] = []
    for event_id, (image_path, frame_objects) in enumerate(grouped.items(), start=1):
        objects = [
            _normalize_object(
                raw_object,
                object_id=f"event-{event_id}-object-{object_index}",
                source_frame=image_path,
                input_json=input_json,
                frame_objects=frame_objects,
            )
            for object_index, raw_object in enumerate(frame_objects, start=1)
        ]
        events.append(
            {
                "event_id": event_id,
                "event_type": "image_detection",
                "start_offset_seconds": None,
                "end_offset_seconds": None,
                "start_video_time": None,
                "end_video_time": None,
                "start_real_time": created_at,
                "end_real_time": created_at,
                "duration_seconds": None,
                "key_frame": image_path or None,
                "evidence_frames": [image_path] if image_path else [],
                "objects": objects,
                "detection_summary": _event_summary(
                    objects,
                    reported_peak_box_count=len(objects),
                    positive_sample_count=1,
                ),
                "risk": _pending_risk(),
            }
        )

    claimed_has_object = bool(detection.get("has_foreign_object", detection.get("has_yiwu", bool(raw_objects))))
    if claimed_has_object != bool(events):
        warnings.append(
            "输入的 has_foreign_object/has_yiwu 与 objects 是否为空不一致，统一结果以 objects 为准。"
        )
    claimed_count = _int(detection.get("num_detections"), len(raw_objects))
    if claimed_count != len(raw_objects):
        warnings.append(
            f"输入 num_detections={claimed_count}，但读取到 {len(raw_objects)} 个有效检测框。"
        )

    class_counts = _clean_count_map(detection.get("class_counts"))
    if not class_counts:
        class_counts = _object_class_counts(
            [item for event in events for item in event["objects"]]
        )

    return _base_document(
        source_type="image",
        source_path=source_path,
        created_at=created_at,
        start_real_time=created_at,
        end_real_time=created_at,
        duration_seconds=None,
        status=str(detection.get("status") or "unknown"),
        events=events,
        detection_box_count=len(raw_objects),
        positive_frame_count=len(events),
        class_counts=class_counts,
        input_format="image_detection_v1",
        input_json=input_json,
        model_path=str(detection.get("model_path") or ""),
        thresholds=_as_dict(detection.get("thresholds")),
        warnings=warnings,
    )


def _normalized_ref(value: Any) -> str:
    return str(value or "").replace("\\", "/").lower()


def _select_event_frame(
    raw_event: Dict[str, Any], detection_frames: Sequence[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    key_frame = _normalized_ref(raw_event.get("key_frame"))
    if key_frame:
        for frame in detection_frames:
            if _normalized_ref(frame.get("image")) == key_frame:
                return frame

    allowed_frames = {_normalized_ref(value) for value in _as_string_list(raw_event.get("frame_images"))}
    candidates = [
        frame for frame in detection_frames if _normalized_ref(frame.get("image")) in allowed_frames
    ]
    if not candidates:
        start = _float(raw_event.get("start_offset_seconds"), -1.0)
        end = _float(raw_event.get("end_offset_seconds"), -1.0)
        candidates = [
            frame
            for frame in detection_frames
            if start <= _float(frame.get("offset_seconds"), -2.0) <= end
        ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (_int(item.get("object_count")), _float(item.get("max_confidence"))),
    )


def convert_video_detection(detection: Dict[str, Any], input_json: Path) -> Dict[str, Any]:
    raw_events = _as_dict_list(detection.get("events"))
    detection_frames = _as_dict_list(detection.get("detection_frames"))
    source_path = str(detection.get("video") or detection.get("source") or "")
    created_at = str(detection.get("created_at") or datetime.now().isoformat(timespec="seconds"))
    warnings: List[str] = []
    events: List[Dict[str, Any]] = []

    for fallback_event_id, raw_event in enumerate(raw_events, start=1):
        event_id = max(1, _int(raw_event.get("event_id"), fallback_event_id))
        selected_frame = _select_event_frame(raw_event, detection_frames)
        key_frame = str(raw_event.get("key_frame") or (selected_frame or {}).get("image") or "")
        frame_objects = _as_dict_list((selected_frame or {}).get("objects"))
        raw_tracks = _as_dict_list(raw_event.get("tracks"))
        representative_inputs: List[Tuple[Dict[str, Any], str]] = []
        for raw_track in raw_tracks:
            representative = _as_dict(raw_track.get("representative_object"))
            if not representative:
                continue
            representative = dict(representative)
            representative["track_id"] = raw_track.get("track_id")
            representative_inputs.append(
                (representative, str(raw_track.get("representative_frame") or key_frame))
            )
        if not representative_inputs:
            representative_inputs = [(raw_object, key_frame) for raw_object in frame_objects]

        objects = []
        for object_index, (raw_object, source_frame) in enumerate(
            representative_inputs, start=1
        ):
            normalized = _normalize_object(
                raw_object,
                object_id=f"event-{event_id}-object-{object_index}",
                source_frame=source_frame,
                input_json=input_json,
                frame_objects=[item[0] for item in representative_inputs],
            )
            objects.append(normalized)
        if selected_frame is None:
            warnings.append(
                f"视频事件 {event_id} 未找到对应检测帧，事件会保留，但 objects 为空。"
            )

        start_offset = _optional_float(raw_event.get("start_offset_seconds"))
        end_offset = _optional_float(raw_event.get("end_offset_seconds"))
        duration = None
        if start_offset is not None and end_offset is not None:
            duration = round(max(0.0, end_offset - start_offset), 3)

        evidence_frames = _as_string_list(raw_event.get("frame_images"))
        for key_frame_item in _as_dict_list(raw_event.get("key_frames")):
            key_frame_image = str(key_frame_item.get("image") or "")
            if key_frame_image and key_frame_image not in evidence_frames:
                evidence_frames.append(key_frame_image)
        if key_frame and key_frame not in evidence_frames:
            evidence_frames.insert(0, key_frame)

        event_class_counts = _clean_count_map(raw_event.get("class_counts"))
        events.append(
            {
                "event_id": event_id,
                "event_type": "video_detection",
                "start_offset_seconds": start_offset,
                "end_offset_seconds": end_offset,
                "start_video_time": (
                    str(raw_event["start_video_time"]) if raw_event.get("start_video_time") is not None else None
                ),
                "end_video_time": (
                    str(raw_event["end_video_time"]) if raw_event.get("end_video_time") is not None else None
                ),
                "start_real_time": (
                    str(raw_event["start_real_time"]) if raw_event.get("start_real_time") is not None else None
                ),
                "end_real_time": (
                    str(raw_event["end_real_time"]) if raw_event.get("end_real_time") is not None else None
                ),
                "duration_seconds": duration,
                "key_frame": key_frame or None,
                "evidence_frames": evidence_frames,
                "objects": objects,
                "detection_summary": _event_summary(
                    objects,
                    reported_peak_box_count=max(
                        0,
                        _int(
                            raw_event.get(
                                "max_simultaneous_objects", raw_event.get("object_count")
                            ),
                            len(objects),
                        ),
                    ),
                    positive_sample_count=max(0, _int(raw_event.get("positive_sample_count"), 1)),
                    unique_object_count=max(
                        0, _int(raw_event.get("unique_object_count"), len(objects))
                    ),
                    track_ids=(
                        raw_event.get("track_ids")
                        if isinstance(raw_event.get("track_ids"), list)
                        else [obj.get("track_id") for obj in objects]
                    ),
                    class_counts=event_class_counts,
                    observed_classes=_as_string_list(raw_event.get("observed_classes")),
                    max_confidence=_optional_float(raw_event.get("max_confidence")),
                ),
                "risk": _pending_risk(),
            }
        )

    claimed_has_object = bool(detection.get("has_foreign_object", bool(raw_events)))
    if claimed_has_object != bool(events):
        warnings.append(
            "输入的 has_foreign_object 与 events 是否为空不一致，统一结果以 events 为准。"
        )
    claimed_event_count = _int(detection.get("num_events"), len(raw_events))
    if claimed_event_count != len(raw_events):
        warnings.append(
            f"输入 num_events={claimed_event_count}，但读取到 {len(raw_events)} 个有效事件。"
        )

    return _base_document(
        source_type="video",
        source_path=source_path,
        created_at=created_at,
        start_real_time=(
            str(detection["video_start_time"]) if detection.get("video_start_time") is not None else None
        ),
        end_real_time=(
            str(detection["video_end_time"]) if detection.get("video_end_time") is not None else None
        ),
        duration_seconds=_optional_float(detection.get("duration_seconds")),
        status=str(detection.get("status") or "unknown"),
        events=events,
        detection_box_count=max(0, _int(detection.get("num_detection_boxes"))),
        positive_frame_count=max(0, _int(detection.get("positive_frames"), len(detection_frames))),
        class_counts=_clean_count_map(detection.get("class_counts")),
        input_format="video_detection_v1",
        input_json=input_json,
        model_path=str(detection.get("model_path") or ""),
        thresholds=_as_dict(detection.get("thresholds")),
        warnings=warnings,
    )


def detect_source_type(detection: Dict[str, Any]) -> str:
    if "events" in detection or "detection_frames" in detection or "video" in detection:
        return "video"
    if "objects" in detection or "num_images" in detection or "has_yiwu" in detection:
        return "image"
    raise ValueError("无法自动判断检测结果是图片还是视频，请使用 --source-type 指定。")


def convert_detection(
    detection: Dict[str, Any], input_json: Path, source_type: str = "auto"
) -> Dict[str, Any]:
    effective_type = detect_source_type(detection) if source_type == "auto" else source_type
    if effective_type == "image":
        document = convert_image_detection(detection, input_json)
    elif effective_type == "video":
        document = convert_video_detection(detection, input_json)
    else:
        raise ValueError(f"不支持的检测类型：{effective_type}")
    assert_valid_unified_alarm(document)
    return document


def validate_unified_alarm(document: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_top = {
        "schema_version",
        "report_id",
        "created_at",
        "source",
        "detection_summary",
        "events",
        "overall_risk",
        "generated_report",
        "provenance",
    }
    missing = sorted(required_top - set(document))
    if missing:
        errors.append(f"缺少顶层字段：{', '.join(missing)}")
        return errors
    if document.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version 必须是 {SCHEMA_VERSION}")

    source = _as_dict(document.get("source"))
    source_type = source.get("type")
    if source_type not in {"image", "video"}:
        errors.append("source.type 必须是 image 或 video")

    events_value = document.get("events")
    if not isinstance(events_value, list):
        errors.append("events 必须是数组")
        return errors
    events = _as_dict_list(events_value)
    if len(events) != len(events_value):
        errors.append("events 中每一项都必须是对象")

    summary = _as_dict(document.get("detection_summary"))
    if _int(summary.get("event_count"), -1) != len(events):
        errors.append("detection_summary.event_count 必须等于 events 数量")
    if bool(summary.get("has_foreign_object")) != bool(events):
        errors.append("detection_summary.has_foreign_object 必须与 events 是否为空一致")
    for count_field in ("event_count", "detection_box_count", "positive_frame_count"):
        value = summary.get(count_field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"detection_summary.{count_field} 必须是非负整数")

    event_ids = []
    for event_index, event in enumerate(events, start=1):
        prefix = f"events[{event_index - 1}]"
        event_id = event.get("event_id")
        if not isinstance(event_id, int) or isinstance(event_id, bool) or event_id < 1:
            errors.append(f"{prefix}.event_id 必须是正整数")
        event_ids.append(event_id)
        expected_event_type = f"{source_type}_detection"
        if event.get("event_type") != expected_event_type:
            errors.append(f"{prefix}.event_type 必须是 {expected_event_type}")
        objects_value = event.get("objects")
        if not isinstance(objects_value, list):
            errors.append(f"{prefix}.objects 必须是数组")
            continue
        event_summary = _as_dict(event.get("detection_summary"))
        if _int(event_summary.get("detection_box_count"), -1) != len(objects_value):
            errors.append(f"{prefix}.detection_summary.detection_box_count 必须等于 objects 数量")
        unique_count = event_summary.get("unique_object_count")
        if not isinstance(unique_count, int) or isinstance(unique_count, bool) or unique_count < 0:
            errors.append(f"{prefix}.detection_summary.unique_object_count 必须是非负整数")
        if not isinstance(event_summary.get("track_ids"), list):
            errors.append(f"{prefix}.detection_summary.track_ids 必须是数组")
        start = _optional_float(event.get("start_offset_seconds"))
        end = _optional_float(event.get("end_offset_seconds"))
        if start is not None and end is not None and end < start:
            errors.append(f"{prefix} 的结束时间不能早于开始时间")
        object_ids = []
        for object_index, detected_object in enumerate(_as_dict_list(objects_value), start=1):
            object_prefix = f"{prefix}.objects[{object_index - 1}]"
            object_ids.append(detected_object.get("object_id"))
            bbox = detected_object.get("bbox_xyxy")
            if not isinstance(bbox, list) or len(bbox) != 4 or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool) for value in bbox
            ):
                errors.append(f"{object_prefix}.bbox_xyxy 必须包含4个数值")
            confidence = detected_object.get("confidence")
            if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
                errors.append(f"{object_prefix}.confidence 必须在 0 到 1 之间")
            area = detected_object.get("area")
            if not isinstance(area, (int, float)) or isinstance(area, bool) or area < 0:
                errors.append(f"{object_prefix}.area 必须是非负数")
        if len(object_ids) != len(set(object_ids)):
            errors.append(f"{prefix} 内 object_id 必须唯一")

    if len(event_ids) != len(set(event_ids)):
        errors.append("event_id 必须唯一")
    if _as_dict(document.get("overall_risk")).get("status") not in {"pending", "completed"}:
        errors.append("overall_risk.status 非法")
    if _as_dict(document.get("generated_report")).get("status") not in {
        "pending",
        "completed",
        "failed",
    }:
        errors.append("generated_report.status 非法")
    return errors


def assert_valid_unified_alarm(document: Dict[str, Any]) -> None:
    errors = validate_unified_alarm(document)
    if errors:
        formatted = "\n".join(f"- {error}" for error in errors)
        raise UnifiedAlarmValidationError(f"统一报警结构验证失败：\n{formatted}")


def convert_file(input_json: Path, output_json: Path, source_type: str = "auto") -> Dict[str, Any]:
    detection = _read_json(input_json)
    document = convert_detection(detection, input_json=input_json, source_type=source_type)
    _write_json(output_json, document)
    return document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将图片或视频检测 JSON 转换为统一报警事件 JSON")
    parser.add_argument("--input", type=Path, required=True, help="图片 detection.json 或视频 detection_results.json")
    parser.add_argument("--output", type=Path, help="统一报警 JSON 输出路径")
    parser.add_argument(
        "--source-type",
        choices=["auto", "image", "video"],
        default="auto",
        help="输入类型，默认自动识别",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.input.with_name(f"{args.input.stem}.unified_alarm.json")
    try:
        document = convert_file(args.input, output, source_type=args.source_type)
    except (FileNotFoundError, ValueError, UnifiedAlarmValidationError) as exc:
        print(exc)
        raise SystemExit(1) from exc

    summary = document["detection_summary"]
    print("统一报警结构转换并验证成功")
    print(f"输入类型：{document['source']['type']}")
    print(f"事件数量：{summary['event_count']}")
    print(f"检测框数量：{summary['detection_box_count']}")
    print(f"输出文件：{output.resolve()}")
    if document["provenance"]["conversion_warnings"]:
        print("转换警告：")
        for warning in document["provenance"]["conversion_warnings"]:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
