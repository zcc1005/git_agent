from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2
from ultralytics import YOLO

from project_config import PROJECT_ROOT
from task2_yolo.detect_yolo import (
    VIS_COLORS,
    normalize_class_name,
    validate_four_class_model,
)
from task2_yolo.yolo_config import CLASS_DISPLAY_NAMES, CLASS_NAMES


def parse_video_start_time(value: str) -> datetime:
    value = value.strip()
    if not value:
        raise ValueError("请填写视频开始时间。")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("视频开始时间格式不正确。") from exc


def format_video_time(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _write_detection_frame(image: Any, objects: List[Dict[str, Any]], output_path: Path) -> None:
    annotated = image.copy()
    for obj in objects:
        x1, y1, x2, y2 = (int(round(value)) for value in obj["bbox_xyxy"])
        class_key = str(obj["class"])
        color = VIS_COLORS.get(class_key, (0, 255, 0))
        label = f"{class_key} {obj['confidence']:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )

    success, encoded = cv2.imencode(".jpg", annotated)
    if not success:
        raise RuntimeError(f"无法编码检测结果图片：{output_path}")
    encoded.tofile(str(output_path))


def _result_objects(result: Any, known_conf: float) -> List[Dict[str, Any]]:
    objects: List[Dict[str, Any]] = []
    boxes = result.boxes
    if boxes is None:
        return objects

    result_names = getattr(result, "names", None)
    for box in boxes:
        predicted_class_id = int(box.cls[0].item())
        predicted_key, predicted_name = normalize_class_name(predicted_class_id, result_names)
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        rejected_as_unknown = confidence < known_conf
        objects.append(
            {
                "class_id": None if rejected_as_unknown else predicted_class_id,
                "class": "unknown" if rejected_as_unknown else predicted_key,
                "class_name": (
                    CLASS_DISPLAY_NAMES["unknown"] if rejected_as_unknown else predicted_name
                ),
                "confidence": round(confidence, 4),
                "rejected_as_unknown": rejected_as_unknown,
                "predicted_class": predicted_key,
                "predicted_class_name": predicted_name,
                "bbox_xyxy": [round(float(value), 2) for value in (x1, y1, x2, y2)],
            }
        )
    return objects


def _peak_class_counts(frames: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    peak: Dict[str, int] = {}
    for frame in frames:
        for name, count in frame["class_counts"].items():
            peak[name] = max(peak.get(name, 0), int(count))
    return peak


@lru_cache(maxsize=1)
def _load_model(model_path: str) -> YOLO:
    model = YOLO(model_path)
    validate_four_class_model(model)
    return model


def _build_event(
    event_id: int,
    frames: List[Dict[str, Any]],
    video_start: datetime,
    sample_period: float,
    duration: float,
) -> Dict[str, Any]:
    start_offset = float(frames[0]["offset_seconds"])
    end_offset = min(duration, float(frames[-1]["offset_seconds"]) + sample_period)
    key_frame = max(
        frames,
        key=lambda item: (item["object_count"], item["max_confidence"]),
    )
    representative_counts = dict(key_frame["class_counts"])
    observed_classes = list(_peak_class_counts(frames))
    return {
        "event_id": event_id,
        "start_offset_seconds": round(start_offset, 3),
        "end_offset_seconds": round(end_offset, 3),
        "start_video_time": format_video_time(start_offset),
        "end_video_time": format_video_time(end_offset),
        "start_real_time": (video_start + timedelta(seconds=start_offset)).isoformat(
            sep=" ", timespec="seconds"
        ),
        "end_real_time": (video_start + timedelta(seconds=end_offset)).isoformat(
            sep=" ", timespec="seconds"
        ),
        "object_count": max(frame["object_count"] for frame in frames),
        "class_counts": representative_counts,
        "classes": list(representative_counts),
        "observed_classes": observed_classes,
        "max_confidence": max(frame["max_confidence"] for frame in frames),
        "positive_sample_count": len(frames),
        "key_frame": key_frame["image"],
        "frame_images": [frame["image"] for frame in frames],
    }


def merge_detection_events(
    positive_frames: List[Dict[str, Any]],
    video_start: datetime,
    sample_fps: float,
    duration: float,
) -> List[Dict[str, Any]]:
    if not positive_frames:
        return []

    sample_period = 1.0 / sample_fps
    max_gap = max(1.5, sample_period * 2.5)
    grouped: List[List[Dict[str, Any]]] = [[positive_frames[0]]]
    for frame in positive_frames[1:]:
        previous = grouped[-1][-1]
        if frame["offset_seconds"] - previous["offset_seconds"] <= max_gap:
            grouped[-1].append(frame)
        else:
            grouped.append([frame])

    return [
        _build_event(index, frames, video_start, sample_period, duration)
        for index, frames in enumerate(grouped, 1)
    ]


def detect_video_foreign_objects(
    video_path: Path,
    model_path: Path,
    output_dir: Path,
    video_start: datetime,
    sample_fps: float = 2.0,
    conf: float = 0.15,
    known_conf: float = 0.40,
) -> Dict[str, Any]:
    if not video_path.is_file():
        raise FileNotFoundError(f"找不到上传的视频：{video_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到 YOLO 模型权重：{model_path}")
    if not 0 < sample_fps <= 60:
        raise ValueError("检测 FPS 必须大于 0 且不超过 60。")
    if not 0 <= conf < known_conf <= 1:
        raise ValueError("置信度必须满足 0 <= 最低置信度 < 已知类别置信度 <= 1。")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV 无法打开视频：{video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0:
        source_fps = 25.0
    total_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    duration = total_frames / source_fps if total_frames else 0.0
    effective_sample_fps = min(sample_fps, source_fps)
    frame_interval = source_fps / effective_sample_fps

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "detected_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(str(model_path.resolve()))
    positive_frames: List[Dict[str, Any]] = []
    sampled_frames = 0
    total_detection_boxes = 0
    source_index = 0
    next_frame_at = 0.0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if source_index + 1e-9 < next_frame_at:
                source_index += 1
                continue

            offset_seconds = source_index / source_fps
            sampled_frames += 1
            result = model.predict(
                source=frame,
                conf=conf,
                imgsz=640,
                save=False,
                verbose=False,
            )[0]
            objects = _result_objects(result, known_conf)
            if objects:
                time_ms = int(round(offset_seconds * 1000))
                image_path = frames_dir / f"frame_{sampled_frames:06d}_{time_ms:010d}ms.jpg"
                _write_detection_frame(frame, objects, image_path)
                class_counts = dict(Counter(obj["class_name"] for obj in objects))
                positive_frames.append(
                    {
                        "frame_index": source_index,
                        "offset_seconds": round(offset_seconds, 3),
                        "video_time": format_video_time(offset_seconds),
                        "real_time": (video_start + timedelta(seconds=offset_seconds)).isoformat(
                            sep=" ", timespec="seconds"
                        ),
                        "object_count": len(objects),
                        "class_counts": class_counts,
                        "max_confidence": max(obj["confidence"] for obj in objects),
                        "image": _project_path(image_path),
                        "objects": objects,
                    }
                )
                total_detection_boxes += len(objects)

            next_frame_at += frame_interval
            source_index += 1
    finally:
        capture.release()

    if duration <= 0:
        duration = source_index / source_fps
    events = merge_detection_events(
        positive_frames,
        video_start,
        effective_sample_fps,
        duration,
    )
    result_data: Dict[str, Any] = {
        "status": "completed",
        "created_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "video": _project_path(video_path),
        "video_start_time": video_start.isoformat(sep=" ", timespec="seconds"),
        "video_end_time": (video_start + timedelta(seconds=duration)).isoformat(
            sep=" ", timespec="seconds"
        ),
        "duration_seconds": round(duration, 3),
        "source_fps": round(source_fps, 3),
        "requested_sample_fps": sample_fps,
        "sample_fps": round(effective_sample_fps, 3),
        "sampled_frames": sampled_frames,
        "positive_frames": len(positive_frames),
        "saved_images": len(positive_frames),
        "num_detection_boxes": total_detection_boxes,
        "has_foreign_object": bool(events),
        "num_events": len(events),
        "class_counts": _peak_class_counts(positive_frames),
        "class_names": CLASS_NAMES,
        "class_display_names": CLASS_DISPLAY_NAMES,
        "events": events,
        "detection_frames": positive_frames,
        "model_path": _project_path(model_path),
        "thresholds": {
            "detection_min_confidence": conf,
            "known_class_min_confidence": known_conf,
        },
    }
    result_path = output_dir / "detection_results.json"
    result_data["result_json"] = _project_path(result_path)
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(result_data, file, ensure_ascii=False, indent=2)
    return result_data
