from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_LOCAL_PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault(
    "YOLO_CONFIG_DIR", str(_LOCAL_PROJECT_ROOT / "outputs" / "ultralytics_runtime")
)

import cv2
from ultralytics import YOLO

from project_config import PROJECT_ROOT
from task2_yolo.detect_yolo import (
    VIS_COLORS,
    normalize_class_name,
    validate_four_class_model,
)
from task2_yolo.postprocess import filter_duplicate_objects as shared_duplicate_filter
from task2_yolo.yolo_config import CLASS_DISPLAY_NAMES, CLASS_NAMES


DEFAULT_IMGSZ = 800
DEFAULT_NMS_IOU = 0.40
DEFAULT_DUPLICATE_IOU = 0.45
DEFAULT_DUPLICATE_CONTAINMENT = 0.80
DEFAULT_CROSS_CLASS_IOU = 0.70
DEFAULT_CROSS_CLASS_CONTAINMENT = 0.92
DEFAULT_EVENT_SILENCE_SECONDS = 1.0
DEFAULT_TRACK_MAX_AGE_SECONDS = 1.0
DEFAULT_TRACK_IOU = 0.15
DEFAULT_TRACK_CENTER_DISTANCE_RATIO = 3.0
DEFAULT_MIN_UNKNOWN_HITS = 2
DEFAULT_UNKNOWN_SINGLE_FRAME_CONF = 0.40


def parse_video_start_time(value: str) -> datetime:
    value = value.strip()
    if not value:
        raise ValueError("请填写视频开始时间。")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("视频开始时间格式不正确。") from exc


def parse_roi(value: str) -> Optional[Tuple[int, int, int, int]]:
    value = value.strip()
    if not value:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI 格式应为 x1,y1,x2,y2，例如 100,50,1180,700。")
    try:
        x1, y1, x2, y2 = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("ROI 坐标必须是整数。") from exc
    if min(x1, y1) < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError("ROI 必须满足 x1>=0、y1>=0、x2>x1、y2>y1。")
    return x1, y1, x2, y2


def format_video_time(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def sample_source_index(
    sample_number: int, source_fps: float, sample_fps: float
) -> int:
    """Return a deterministic source-frame index for a zero-based sample number.

    With the same source FPS, integer-multiple sampling rates share source indices;
    therefore raising 2 FPS to 4 FPS adds intermediate frames without replacing the
    original 2 FPS frames.
    """

    if sample_number < 0 or source_fps <= 0 or sample_fps <= 0:
        raise ValueError("采样序号不能为负，source_fps 和 sample_fps 必须大于0。")
    return int(round(sample_number * source_fps / sample_fps))


def _project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _bbox_area(bbox: Sequence[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(
        0.0, float(bbox[3]) - float(bbox[1])
    )


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    union = _bbox_area(first) + _bbox_area(second) - intersection
    return intersection / union if union > 0 else 0.0


def bbox_containment(first: Sequence[float], second: Sequence[float]) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    smaller = min(_bbox_area(first), _bbox_area(second))
    return intersection / smaller if smaller > 0 else 0.0


def _center(bbox: Sequence[float]) -> Tuple[float, float]:
    return (
        (float(bbox[0]) + float(bbox[2])) / 2.0,
        (float(bbox[1]) + float(bbox[3])) / 2.0,
    )


def _center_distance_ratio(first: Sequence[float], second: Sequence[float]) -> float:
    first_center = _center(first)
    second_center = _center(second)
    distance = math.hypot(
        first_center[0] - second_center[0], first_center[1] - second_center[1]
    )
    first_diagonal = math.hypot(
        float(first[2]) - float(first[0]), float(first[3]) - float(first[1])
    )
    second_diagonal = math.hypot(
        float(second[2]) - float(second[0]), float(second[3]) - float(second[1])
    )
    scale = max(first_diagonal, second_diagonal, 1.0)
    return distance / scale


def _canonical_raw_object(raw_object: Dict[str, Any]) -> Dict[str, Any]:
    predicted_class = str(
        raw_object.get("predicted_class") or raw_object.get("class") or "unknown"
    )
    if predicted_class == "unknown" and raw_object.get("predicted_class"):
        predicted_class = str(raw_object["predicted_class"])
    predicted_name = str(
        raw_object.get("predicted_class_name")
        or raw_object.get("class_name")
        or CLASS_DISPLAY_NAMES.get(predicted_class, predicted_class)
    )
    raw_class_id = raw_object.get("predicted_class_id", raw_object.get("class_id"))
    predicted_class_id = int(raw_class_id) if raw_class_id not in (None, "") else None
    raw_bbox = raw_object.get("bbox_xyxy") or raw_object.get("bbox") or [0, 0, 0, 0]
    bbox = [round(float(value), 2) for value in list(raw_bbox)[:4]]
    if len(bbox) != 4:
        bbox = [0.0, 0.0, 0.0, 0.0]
    return {
        "predicted_class_id": predicted_class_id,
        "predicted_class": predicted_class,
        "predicted_class_name": predicted_name,
        "confidence": round(float(raw_object.get("confidence", 0.0) or 0.0), 4),
        "bbox_xyxy": bbox,
        "area": round(_bbox_area(bbox), 2),
    }


def filter_duplicate_objects(
    raw_objects: Sequence[Dict[str, Any]],
    duplicate_iou: float = DEFAULT_DUPLICATE_IOU,
    containment_threshold: float = DEFAULT_DUPLICATE_CONTAINMENT,
    class_agnostic: bool = False,
    cross_class_iou: float = DEFAULT_CROSS_CLASS_IOU,
    cross_class_containment: float = DEFAULT_CROSS_CLASS_CONTAINMENT,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Second-stage duplicate suppression after Ultralytics NMS.

    Same-class boxes use the normal thresholds. Competing cross-class boxes are
    suppressed only at much stronger overlap/containment thresholds.
    """

    return shared_duplicate_filter(
        raw_objects,
        duplicate_iou=duplicate_iou,
        containment_threshold=containment_threshold,
        class_agnostic=class_agnostic,
        cross_class_iou=cross_class_iou,
        cross_class_containment=cross_class_containment,
    )


@dataclass
class TrackState:
    track_id: int
    predicted_class_id: Optional[int]
    predicted_class: str
    predicted_class_name: str
    bbox_xyxy: List[float]
    first_seen: float
    last_seen: float
    hit_count: int
    consecutive_hits: int
    max_confidence: float
    confirmed: bool = False
    confirmed_class: Optional[str] = None
    confirmed_class_name: Optional[str] = None
    confirmation_reason: Optional[str] = None
    velocity_x: float = 0.0
    velocity_y: float = 0.0

    def predicted_bbox(self, offset_seconds: float) -> List[float]:
        age = max(0.0, offset_seconds - self.last_seen)
        dx = self.velocity_x * age
        dy = self.velocity_y * age
        return [
            self.bbox_xyxy[0] + dx,
            self.bbox_xyxy[1] + dy,
            self.bbox_xyxy[2] + dx,
            self.bbox_xyxy[3] + dy,
        ]


class LightweightTracker:
    def __init__(
        self,
        known_conf: float = 0.40,
        track_max_age_seconds: float = DEFAULT_TRACK_MAX_AGE_SECONDS,
        min_unknown_hits: int = DEFAULT_MIN_UNKNOWN_HITS,
        unknown_single_frame_conf: float = DEFAULT_UNKNOWN_SINGLE_FRAME_CONF,
        track_iou: float = DEFAULT_TRACK_IOU,
        center_distance_ratio: float = DEFAULT_TRACK_CENTER_DISTANCE_RATIO,
    ) -> None:
        self.known_conf = known_conf
        self.track_max_age_seconds = track_max_age_seconds
        self.min_unknown_hits = min_unknown_hits
        self.unknown_single_frame_conf = unknown_single_frame_conf
        self.track_iou = track_iou
        self.center_distance_ratio = center_distance_ratio
        self.next_track_id = 1
        self.active_tracks: Dict[int, TrackState] = {}
        self.all_tracks: Dict[int, TrackState] = {}

    def _new_track(self, detection: Dict[str, Any], offset_seconds: float) -> TrackState:
        confidence = float(detection["confidence"])
        known = confidence >= self.known_conf
        confirmed_candidate = not known and (
            confidence >= self.unknown_single_frame_conf
            or self.min_unknown_hits <= 1
        )
        track = TrackState(
            track_id=self.next_track_id,
            predicted_class_id=detection["predicted_class_id"],
            predicted_class=detection["predicted_class"],
            predicted_class_name=detection["predicted_class_name"],
            bbox_xyxy=list(detection["bbox_xyxy"]),
            first_seen=offset_seconds,
            last_seen=offset_seconds,
            hit_count=1,
            consecutive_hits=1,
            max_confidence=confidence,
            confirmed=known or confirmed_candidate,
            confirmed_class=(
                detection["predicted_class"] if known or confirmed_candidate else None
            ),
            confirmed_class_name=(
                detection["predicted_class_name"]
                if known or confirmed_candidate
                else None
            ),
            confirmation_reason=(
                "known_confidence"
                if known
                else (
                    "single_high_class_candidate"
                    if confidence >= self.unknown_single_frame_conf
                    else ("repeated_class_candidate" if confirmed_candidate else None)
                )
            ),
        )
        self.next_track_id += 1
        self.active_tracks[track.track_id] = track
        self.all_tracks[track.track_id] = track
        return track

    def _match_score(
        self, track: TrackState, detection: Dict[str, Any], offset_seconds: float
    ) -> Optional[float]:
        if detection["predicted_class"] != track.predicted_class:
            return None
        age = offset_seconds - track.last_seen
        if age < 0 or age > self.track_max_age_seconds:
            return None
        predicted_bbox = track.predicted_bbox(offset_seconds)
        overlap = bbox_iou(predicted_bbox, detection["bbox_xyxy"])
        center_ratio = _center_distance_ratio(
            predicted_bbox, detection["bbox_xyxy"]
        )
        if overlap < self.track_iou and center_ratio > self.center_distance_ratio:
            return None
        distance_score = max(
            0.0, 1.0 - center_ratio / max(self.center_distance_ratio, 1e-6)
        )
        age_score = max(
            0.0, 1.0 - age / max(self.track_max_age_seconds, 1e-6)
        )
        return overlap * 2.0 + distance_score * 0.6 + age_score * 0.2

    def _update_track(
        self, track: TrackState, detection: Dict[str, Any], offset_seconds: float
    ) -> None:
        previous_center = _center(track.bbox_xyxy)
        current_center = _center(detection["bbox_xyxy"])
        elapsed = max(1e-6, offset_seconds - track.last_seen)
        measured_velocity_x = (current_center[0] - previous_center[0]) / elapsed
        measured_velocity_y = (current_center[1] - previous_center[1]) / elapsed
        track.velocity_x = track.velocity_x * 0.5 + measured_velocity_x * 0.5
        track.velocity_y = track.velocity_y * 0.5 + measured_velocity_y * 0.5
        if elapsed <= self.track_max_age_seconds:
            track.consecutive_hits += 1
        else:
            track.consecutive_hits = 1
        track.last_seen = offset_seconds
        track.bbox_xyxy = list(detection["bbox_xyxy"])
        track.hit_count += 1
        track.max_confidence = max(track.max_confidence, float(detection["confidence"]))

        if float(detection["confidence"]) >= self.known_conf:
            track.confirmed = True
            track.confirmed_class = detection["predicted_class"]
            track.confirmed_class_name = detection["predicted_class_name"]
            track.confirmation_reason = "known_confidence"
            track.predicted_class_id = detection["predicted_class_id"]
        elif not track.confirmed and float(detection["confidence"]) >= self.unknown_single_frame_conf:
            track.confirmed = True
            track.confirmed_class = detection["predicted_class"]
            track.confirmed_class_name = detection["predicted_class_name"]
            track.predicted_class_id = detection["predicted_class_id"]
            track.confirmation_reason = "single_high_class_candidate"
        elif not track.confirmed and track.consecutive_hits >= self.min_unknown_hits:
            track.confirmed = True
            track.confirmed_class = detection["predicted_class"]
            track.confirmed_class_name = detection["predicted_class_name"]
            track.predicted_class_id = detection["predicted_class_id"]
            track.confirmation_reason = "repeated_class_candidate"

    def _decorate_detection(
        self, detection: Dict[str, Any], track: TrackState, match_type: str
    ) -> Dict[str, Any]:
        confidence = float(detection["confidence"])
        if track.confirmed:
            class_key = str(track.confirmed_class)
            class_name = str(track.confirmed_class_name)
            class_id = track.predicted_class_id if class_key != "unknown" else None
            if track.confirmation_reason == "known_confidence" and confidence >= self.known_conf:
                detection_state = "confirmed_known"
            else:
                detection_state = "confirmed_by_track"
        else:
            class_key = "unknown"
            class_name = CLASS_DISPLAY_NAMES["unknown"]
            class_id = None
            detection_state = "unknown_candidate"
        return {
            **detection,
            "class_id": class_id,
            "class": class_key,
            "class_name": class_name,
            "rejected_as_unknown": class_key == "unknown",
            "detection_state": detection_state,
            "track_id": track.track_id,
            "track_hit_count": track.hit_count,
            "track_confirmed": track.confirmed,
            "confirmation_reason": track.confirmation_reason,
            "tracking_match": match_type,
        }

    def update(
        self, detections: Sequence[Dict[str, Any]], offset_seconds: float
    ) -> List[Dict[str, Any]]:
        for track_id, track in list(self.active_tracks.items()):
            if offset_seconds - track.last_seen > self.track_max_age_seconds:
                del self.active_tracks[track_id]

        candidates: List[Tuple[float, int, int]] = []
        active_items = list(self.active_tracks.items())
        for detection_index, detection in enumerate(detections):
            for track_id, track in active_items:
                score = self._match_score(track, detection, offset_seconds)
                if score is not None:
                    candidates.append((score, detection_index, track_id))
        candidates.sort(reverse=True)

        matched_detections: Dict[int, int] = {}
        used_tracks = set()
        for _, detection_index, track_id in candidates:
            if detection_index in matched_detections or track_id in used_tracks:
                continue
            matched_detections[detection_index] = track_id
            used_tracks.add(track_id)

        tracked: List[Dict[str, Any]] = []
        for detection_index, detection in enumerate(detections):
            track_id = matched_detections.get(detection_index)
            if track_id is None:
                track = self._new_track(detection, offset_seconds)
                match_type = "new_track"
            else:
                track = self.active_tracks[track_id]
                self._update_track(track, detection, offset_seconds)
                match_type = "matched_track"
            tracked.append(self._decorate_detection(detection, track, match_type))
        return tracked


def process_frame_objects(
    raw_objects: Sequence[Dict[str, Any]],
    tracker: LightweightTracker,
    offset_seconds: float,
    duplicate_iou: float = DEFAULT_DUPLICATE_IOU,
    duplicate_containment: float = DEFAULT_DUPLICATE_CONTAINMENT,
    class_agnostic_duplicates: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    deduplicated, ignored = filter_duplicate_objects(
        raw_objects,
        duplicate_iou=duplicate_iou,
        containment_threshold=duplicate_containment,
        class_agnostic=class_agnostic_duplicates,
    )
    tracked_objects = tracker.update(deduplicated, offset_seconds)
    confirmed = [
        item for item in tracked_objects if item["detection_state"] != "unknown_candidate"
    ]
    candidates = [
        item for item in tracked_objects if item["detection_state"] == "unknown_candidate"
    ]
    return {
        "raw_objects": [_canonical_raw_object(item) for item in raw_objects],
        "deduplicated_objects": tracked_objects,
        "ignored_objects": ignored,
        "unknown_candidates": candidates,
        "objects": confirmed,
    }


def _write_detection_frame(
    image: Any,
    confirmed_objects: Sequence[Dict[str, Any]],
    candidate_objects: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    annotated = image.copy()
    for obj in list(confirmed_objects) + list(candidate_objects):
        x1, y1, x2, y2 = (int(round(value)) for value in obj["bbox_xyxy"])
        is_candidate = obj["detection_state"] == "unknown_candidate"
        class_key = str(obj["class"])
        color = (150, 150, 150) if is_candidate else VIS_COLORS.get(class_key, (0, 255, 0))
        prefix = "candidate" if is_candidate else f"track#{obj['track_id']}"
        label = f"{prefix} {class_key} {obj['confidence']:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1 if is_candidate else 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            1 if is_candidate else 2,
        )

    success, encoded = cv2.imencode(".jpg", annotated)
    if not success:
        raise RuntimeError(f"无法编码检测结果图片：{output_path}")
    encoded.tofile(str(output_path))


def _raw_result_objects(
    result: Any, offset_x: int = 0, offset_y: int = 0
) -> List[Dict[str, Any]]:
    objects: List[Dict[str, Any]] = []
    boxes = result.boxes
    if boxes is None:
        return objects
    result_names = getattr(result, "names", None)
    for box in boxes:
        predicted_class_id = int(box.cls[0].item())
        predicted_key, predicted_name = normalize_class_name(
            predicted_class_id, result_names
        )
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        objects.append(
            {
                "predicted_class_id": predicted_class_id,
                "predicted_class": predicted_key,
                "predicted_class_name": predicted_name,
                "confidence": round(confidence, 4),
                "bbox_xyxy": [
                    round(float(x1) + offset_x, 2),
                    round(float(y1) + offset_y, 2),
                    round(float(x2) + offset_x, 2),
                    round(float(y2) + offset_y, 2),
                ],
            }
        )
    return objects


def _frame_class_counts(objects: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(str(item["class_name"]) for item in objects))


def _track_records(frames: Sequence[Dict[str, Any]]) -> Dict[int, List[Tuple[Dict[str, Any], Dict[str, Any]]]]:
    records: Dict[int, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = {}
    for frame in frames:
        for obj in frame["objects"]:
            records.setdefault(int(obj["track_id"]), []).append((frame, obj))
    return records


def _track_summaries(frames: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for track_id, observations in sorted(_track_records(frames).items()):
        representative_frame, representative_object = max(
            observations,
            key=lambda item: (
                item[1]["detection_state"] == "confirmed_known",
                item[1]["confidence"],
            ),
        )
        first_frame = min(observations, key=lambda item: item[0]["offset_seconds"])[0]
        last_frame = max(observations, key=lambda item: item[0]["offset_seconds"])[0]
        summaries.append(
            {
                "track_id": track_id,
                "class_id": representative_object["class_id"],
                "class": representative_object["class"],
                "class_name": representative_object["class_name"],
                "max_confidence": max(item[1]["confidence"] for item in observations),
                "confirmed_observation_count": len(observations),
                "track_hit_count": max(item[1]["track_hit_count"] for item in observations),
                "confirmation_reason": representative_object["confirmation_reason"],
                "first_offset_seconds": first_frame["offset_seconds"],
                "last_offset_seconds": last_frame["offset_seconds"],
                "first_video_time": first_frame["video_time"],
                "last_video_time": last_frame["video_time"],
                "representative_frame": representative_frame["image"],
                "representative_object": dict(representative_object),
            }
        )
    return summaries


def _key_frames(
    frames: Sequence[Dict[str, Any]], track_summaries: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    selected_images = [str(item["representative_frame"]) for item in track_summaries]
    peak_frame = max(
        frames, key=lambda item: (item["object_count"], item["max_confidence"])
    )
    selected_images.append(str(peak_frame["image"]))
    unique_images = list(dict.fromkeys(selected_images))
    frame_lookup = {str(frame["image"]): frame for frame in frames}
    key_frames: List[Dict[str, Any]] = []
    for image in unique_images:
        frame = frame_lookup[image]
        key_frames.append(
            {
                "image": image,
                "video_time": frame["video_time"],
                "real_time": frame["real_time"],
                "track_ids": sorted({int(obj["track_id"]) for obj in frame["objects"]}),
                "object_count": frame["object_count"],
                "class_counts": dict(frame["class_counts"]),
            }
        )
    return key_frames


def _build_event(
    event_id: int,
    frames: List[Dict[str, Any]],
    video_start: datetime,
    sample_period: float,
    duration: float,
) -> Dict[str, Any]:
    start_offset = float(frames[0]["offset_seconds"])
    end_offset = min(duration, float(frames[-1]["offset_seconds"]) + sample_period)
    max_frame = max(
        frames, key=lambda item: (item["object_count"], item["max_confidence"])
    )
    tracks = _track_summaries(frames)
    class_counts = dict(Counter(track["class_name"] for track in tracks))
    key_frames = _key_frames(frames, tracks)
    track_ids = [int(track["track_id"]) for track in tracks]
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
        "object_count": max_frame["object_count"],
        "max_simultaneous_objects": max_frame["object_count"],
        "unique_object_count": len(track_ids),
        "class_counts": class_counts,
        "classes": list(class_counts),
        "observed_classes": list(class_counts),
        "max_confidence": max(frame["max_confidence"] for frame in frames),
        "positive_sample_count": len(frames),
        "track_ids": track_ids,
        "tracks": tracks,
        "key_frame": max_frame["image"],
        "key_frames": key_frames,
        "frame_images": [frame["image"] for frame in frames],
    }


def merge_detection_events(
    positive_frames: List[Dict[str, Any]],
    video_start: datetime,
    sample_fps: float,
    duration: float,
    event_silence_seconds: float = DEFAULT_EVENT_SILENCE_SECONDS,
) -> List[Dict[str, Any]]:
    """Group only frames containing confirmed tracked objects.

    Candidate-only frames are deliberately excluded, so a weak one-frame candidate
    cannot bridge two otherwise separate waves.
    """

    if not positive_frames:
        return []
    sample_period = 1.0 / sample_fps
    grouped: List[List[Dict[str, Any]]] = [[positive_frames[0]]]
    for frame in positive_frames[1:]:
        previous_confirmed = grouped[-1][-1]
        silence = float(frame["offset_seconds"]) - float(
            previous_confirmed["offset_seconds"]
        )
        if silence <= event_silence_seconds + 1e-9:
            grouped[-1].append(frame)
        else:
            grouped.append([frame])
    return [
        _build_event(index, frames, video_start, sample_period, duration)
        for index, frames in enumerate(grouped, 1)
    ]


@lru_cache(maxsize=1)
def _load_model(model_path: str) -> YOLO:
    model = YOLO(model_path)
    validate_four_class_model(model)
    return model


def _validated_roi(
    roi: Optional[Tuple[int, int, int, int]], frame_width: int, frame_height: int
) -> Optional[Tuple[int, int, int, int]]:
    if roi is None:
        return None
    x1, y1, x2, y2 = roi
    if x2 > frame_width or y2 > frame_height:
        raise ValueError(
            f"ROI {roi} 超出视频尺寸 {frame_width}x{frame_height}。"
        )
    return roi


def _validate_parameters(
    sample_fps: float,
    conf: float,
    known_conf: float,
    imgsz: int,
    nms_iou: float,
    duplicate_iou: float,
    event_silence_seconds: float,
    track_max_age_seconds: float,
    min_unknown_hits: int,
    unknown_single_frame_conf: float,
) -> None:
    if not 0 < sample_fps <= 60:
        raise ValueError("检测 FPS 必须大于 0 且不超过 60。")
    if not 0 <= conf < known_conf <= 1:
        raise ValueError("置信度必须满足 0 <= 最低置信度 < 已知类别置信度 <= 1。")
    if not conf <= unknown_single_frame_conf <= known_conf:
        raise ValueError("单帧候选确认阈值必须位于最低置信度和类别确认阈值之间。")
    if imgsz < 32:
        raise ValueError("推理尺寸 imgsz 必须不小于 32。")
    if not 0 < nms_iou < 1 or not 0 < duplicate_iou < 1:
        raise ValueError("NMS IoU 和二次去重 IoU 必须位于 0 到 1 之间。")
    if event_silence_seconds <= 0 or track_max_age_seconds < 0:
        raise ValueError("事件静默时间必须大于0，轨迹最大失联时间不能为负数。")
    if min_unknown_hits < 1:
        raise ValueError("min_unknown_hits 必须至少为1。")


def detect_video_foreign_objects(
    video_path: Path,
    model_path: Path,
    output_dir: Path,
    video_start: datetime,
    sample_fps: float = 4.0,
    conf: float = 0.25,
    known_conf: float = 0.40,
    imgsz: int = DEFAULT_IMGSZ,
    nms_iou: float = DEFAULT_NMS_IOU,
    agnostic_nms: bool = False,
    duplicate_iou: float = DEFAULT_DUPLICATE_IOU,
    duplicate_containment: float = DEFAULT_DUPLICATE_CONTAINMENT,
    event_silence_seconds: float = DEFAULT_EVENT_SILENCE_SECONDS,
    track_max_age_seconds: float = DEFAULT_TRACK_MAX_AGE_SECONDS,
    min_unknown_hits: int = DEFAULT_MIN_UNKNOWN_HITS,
    unknown_single_frame_conf: float = DEFAULT_UNKNOWN_SINGLE_FRAME_CONF,
    track_iou: float = DEFAULT_TRACK_IOU,
    track_center_distance_ratio: float = DEFAULT_TRACK_CENTER_DISTANCE_RATIO,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> Dict[str, Any]:
    if not video_path.is_file():
        raise FileNotFoundError(f"找不到上传的视频：{video_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到 YOLO 模型权重：{model_path}")
    _validate_parameters(
        sample_fps,
        conf,
        known_conf,
        imgsz,
        nms_iou,
        duplicate_iou,
        event_silence_seconds,
        track_max_age_seconds,
        min_unknown_hits,
        unknown_single_frame_conf,
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV 无法打开视频：{video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0:
        source_fps = 25.0
    total_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    duration = total_frames / source_fps if total_frames else 0.0
    effective_sample_fps = min(sample_fps, source_fps)

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "detected_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    model = _load_model(str(model_path.resolve()))
    tracker = LightweightTracker(
        known_conf=known_conf,
        track_max_age_seconds=track_max_age_seconds,
        min_unknown_hits=min_unknown_hits,
        unknown_single_frame_conf=unknown_single_frame_conf,
        track_iou=track_iou,
        center_distance_ratio=track_center_distance_ratio,
    )

    detection_frames: List[Dict[str, Any]] = []
    positive_frames: List[Dict[str, Any]] = []
    sampled_frames = 0
    source_index = 0
    next_frame_index = 0
    total_raw_boxes = 0
    total_deduplicated_boxes = 0
    total_confirmed_boxes = 0
    total_ignored_boxes = 0
    candidate_frame_count = 0
    effective_roi: Optional[Tuple[int, int, int, int]] = None

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if source_index < next_frame_index:
                source_index += 1
                continue

            if sampled_frames == 0:
                effective_roi = _validated_roi(roi, frame.shape[1], frame.shape[0])
            offset_seconds = source_index / source_fps
            sampled_frames += 1
            next_frame_index = sample_source_index(
                sampled_frames, source_fps, effective_sample_fps
            )

            if effective_roi is None:
                inference_frame = frame
                offset_x = offset_y = 0
            else:
                x1, y1, x2, y2 = effective_roi
                inference_frame = frame[y1:y2, x1:x2]
                offset_x, offset_y = x1, y1

            result = model.predict(
                source=inference_frame,
                conf=conf,
                iou=nms_iou,
                imgsz=imgsz,
                agnostic_nms=agnostic_nms,
                save=False,
                verbose=False,
            )[0]
            raw_objects = _raw_result_objects(result, offset_x=offset_x, offset_y=offset_y)
            processed = process_frame_objects(
                raw_objects,
                tracker=tracker,
                offset_seconds=offset_seconds,
                duplicate_iou=duplicate_iou,
                duplicate_containment=duplicate_containment,
                class_agnostic_duplicates=agnostic_nms,
            )
            total_raw_boxes += len(processed["raw_objects"])
            total_deduplicated_boxes += len(processed["deduplicated_objects"])
            total_confirmed_boxes += len(processed["objects"])
            total_ignored_boxes += len(processed["ignored_objects"])

            if processed["raw_objects"]:
                time_ms = int(round(offset_seconds * 1000))
                image_path = frames_dir / f"frame_{sampled_frames:06d}_{time_ms:010d}ms.jpg"
                _write_detection_frame(
                    frame,
                    confirmed_objects=processed["objects"],
                    candidate_objects=processed["unknown_candidates"],
                    output_path=image_path,
                )
                confirmed_objects = processed["objects"]
                class_counts = _frame_class_counts(confirmed_objects)
                frame_data = {
                    "frame_index": source_index,
                    "offset_seconds": round(offset_seconds, 3),
                    "video_time": format_video_time(offset_seconds),
                    "real_time": (video_start + timedelta(seconds=offset_seconds)).isoformat(
                        sep=" ", timespec="seconds"
                    ),
                    "raw_object_count": len(processed["raw_objects"]),
                    "deduplicated_object_count": len(processed["deduplicated_objects"]),
                    "ignored_object_count": len(processed["ignored_objects"]),
                    "candidate_count": len(processed["unknown_candidates"]),
                    "object_count": len(confirmed_objects),
                    "class_counts": class_counts,
                    "max_confidence": (
                        max(obj["confidence"] for obj in confirmed_objects)
                        if confirmed_objects
                        else 0.0
                    ),
                    "image": _project_path(image_path),
                    "raw_objects": processed["raw_objects"],
                    "ignored_objects": processed["ignored_objects"],
                    "deduplicated_objects": processed["deduplicated_objects"],
                    "unknown_candidates": processed["unknown_candidates"],
                    "objects": confirmed_objects,
                }
                detection_frames.append(frame_data)
                if processed["unknown_candidates"]:
                    candidate_frame_count += 1
                if confirmed_objects:
                    positive_frames.append(frame_data)
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
        event_silence_seconds=event_silence_seconds,
    )
    all_event_tracks = {
        int(track["track_id"]): track
        for event in events
        for track in event["tracks"]
    }
    class_counts = dict(
        Counter(track["class_name"] for track in all_event_tracks.values())
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
        "raw_detection_frames": len(detection_frames),
        "positive_frames": len(positive_frames),
        "candidate_frames": candidate_frame_count,
        "saved_images": len(detection_frames),
        "num_raw_detection_boxes": total_raw_boxes,
        "num_deduplicated_boxes": total_deduplicated_boxes,
        "num_ignored_duplicate_boxes": total_ignored_boxes,
        "num_detection_boxes": total_confirmed_boxes,
        "unique_object_count": len(all_event_tracks),
        "has_foreign_object": bool(events),
        "num_events": len(events),
        "class_counts": class_counts,
        "class_names": CLASS_NAMES,
        "class_display_names": CLASS_DISPLAY_NAMES,
        "events": events,
        "tracks": list(all_event_tracks.values()),
        "detection_frames": detection_frames,
        "model_path": _project_path(model_path),
        "thresholds": {
            "detection_min_confidence": conf,
            "known_class_min_confidence": known_conf,
            "unknown_single_frame_confidence": unknown_single_frame_conf,
            "min_unknown_hits": min_unknown_hits,
            "nms_iou": nms_iou,
            "duplicate_iou": duplicate_iou,
            "duplicate_containment": duplicate_containment,
            "track_iou": track_iou,
            "track_center_distance_ratio": track_center_distance_ratio,
        },
        "temporal_parameters": {
            "event_silence_seconds": event_silence_seconds,
            "track_max_age_seconds": track_max_age_seconds,
        },
        "inference_parameters": {
            "imgsz": imgsz,
            "agnostic_nms": agnostic_nms,
            "roi": list(effective_roi) if effective_roi else None,
            "sampling_rule": "round(sample_index * source_fps / sample_fps)",
        },
        "notes": {
            "raw_objects": "Ultralytics NMS 后、应用层二次去重前的模型框",
            "ignored_objects": "应用层判定为重复/包含关系的调试框",
            "unknown_candidates": "兼容字段名：未达到确认规则的候选框，不触发报警、不延长事件",
            "objects": "已确认并带 track_id 的正式报警框",
        },
    }
    result_path = output_dir / "detection_results.json"
    result_data["result_json"] = _project_path(result_path)
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(result_data, file, ensure_ascii=False, indent=2)
    return result_data
