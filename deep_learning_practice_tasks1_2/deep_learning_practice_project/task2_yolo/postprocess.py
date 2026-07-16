from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


def bbox_area(bbox: Sequence[float]) -> float:
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
    union = bbox_area(first) + bbox_area(second) - intersection
    return intersection / union if union > 0 else 0.0


def bbox_containment(first: Sequence[float], second: Sequence[float]) -> float:
    """Return the intersection divided by the smaller box area."""

    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    smaller = min(bbox_area(first), bbox_area(second))
    return intersection / smaller if smaller > 0 else 0.0


def canonical_object(raw_object: Dict[str, Any]) -> Dict[str, Any]:
    predicted_class = str(
        raw_object.get("predicted_class") or raw_object.get("class") or "unknown"
    )
    predicted_name = str(
        raw_object.get("predicted_class_name")
        or raw_object.get("class_name")
        or predicted_class
    )
    raw_class_id = raw_object.get("predicted_class_id", raw_object.get("class_id"))
    predicted_class_id = int(raw_class_id) if raw_class_id not in (None, "") else None
    raw_bbox = raw_object.get("bbox_xyxy") or raw_object.get("bbox") or [0, 0, 0, 0]
    bbox = [round(float(value), 2) for value in list(raw_bbox)[:4]]
    if len(bbox) != 4:
        bbox = [0.0, 0.0, 0.0, 0.0]
    return {
        **raw_object,
        "predicted_class_id": predicted_class_id,
        "predicted_class": predicted_class,
        "predicted_class_name": predicted_name,
        "confidence": round(float(raw_object.get("confidence", 0.0) or 0.0), 4),
        "bbox_xyxy": bbox,
        "area": round(bbox_area(bbox), 2),
    }


def filter_duplicate_objects(
    raw_objects: Sequence[Dict[str, Any]],
    duplicate_iou: float = 0.45,
    containment_threshold: float = 0.80,
    class_agnostic: bool = False,
    cross_class_iou: Optional[float] = 0.70,
    cross_class_containment: Optional[float] = 0.92,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Suppress duplicate boxes after model NMS.

    Same-class boxes use the regular thresholds. Different-class boxes are only
    suppressed when they overlap much more strongly, which removes competing
    labels for one physical target without deleting ordinary adjacent objects.
    """

    ordered = sorted(
        (canonical_object(item) for item in raw_objects),
        key=lambda item: float(item.get("selection_score", item["confidence"])),
        reverse=True,
    )
    kept: List[Dict[str, Any]] = []
    ignored: List[Dict[str, Any]] = []
    for candidate in ordered:
        duplicate_of: Optional[Dict[str, Any]] = None
        duplicate_reason = ""
        for accepted in kept:
            same_class = candidate["predicted_class"] == accepted["predicted_class"]
            overlap = bbox_iou(candidate["bbox_xyxy"], accepted["bbox_xyxy"])
            containment = bbox_containment(
                candidate["bbox_xyxy"], accepted["bbox_xyxy"]
            )

            if same_class or class_agnostic:
                if overlap >= duplicate_iou:
                    duplicate_of = accepted
                    duplicate_reason = f"duplicate_iou={overlap:.3f}"
                    break
                if containment >= containment_threshold:
                    duplicate_of = accepted
                    duplicate_reason = f"duplicate_containment={containment:.3f}"
                    break
            else:
                strong_cross_class_overlap = (
                    cross_class_iou is not None and overlap >= cross_class_iou
                )
                strong_cross_class_containment = (
                    cross_class_containment is not None
                    and containment >= cross_class_containment
                )
                if strong_cross_class_overlap or strong_cross_class_containment:
                    duplicate_of = accepted
                    metric = "iou" if strong_cross_class_overlap else "containment"
                    value = overlap if strong_cross_class_overlap else containment
                    duplicate_reason = f"cross_class_{metric}={value:.3f}"
                    break

        if duplicate_of is None:
            kept.append(candidate)
        else:
            ignored.append(
                {
                    **candidate,
                    "detection_state": "background_ignored",
                    "filter_reason": duplicate_reason,
                    "kept_class": duplicate_of["predicted_class"],
                    "kept_confidence": duplicate_of["confidence"],
                }
            )
    return kept, ignored


def filter_implausible_geometry(
    objects: Sequence[Dict[str, Any]],
    image_width: int,
    image_height: int,
    max_area_ratio: float = 0.65,
    large_edge_area_ratio: float = 0.35,
    edge_margin_ratio: float = 0.01,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Conservatively remove huge/background-like boxes.

    The edge rule only fires for a large box touching at least two image edges,
    so an ordinary object near one side of the belt is retained.
    """

    image_area = max(1.0, float(image_width * image_height))
    margin_x = image_width * edge_margin_ratio
    margin_y = image_height * edge_margin_ratio
    kept: List[Dict[str, Any]] = []
    ignored: List[Dict[str, Any]] = []
    for raw_object in objects:
        item = canonical_object(raw_object)
        x1, y1, x2, y2 = item["bbox_xyxy"]
        area_ratio = float(item["area"]) / image_area
        edge_count = sum(
            (
                x1 <= margin_x,
                y1 <= margin_y,
                x2 >= image_width - margin_x,
                y2 >= image_height - margin_y,
            )
        )
        reason = ""
        if area_ratio >= max_area_ratio:
            reason = f"box_area_ratio={area_ratio:.3f}"
        elif area_ratio >= large_edge_area_ratio and edge_count >= 2:
            reason = f"large_edge_box(area_ratio={area_ratio:.3f},edges={edge_count})"

        if reason:
            ignored.append(
                {
                    **item,
                    "detection_state": "background_ignored",
                    "filter_reason": reason,
                }
            )
        else:
            kept.append(item)
    return kept, ignored
