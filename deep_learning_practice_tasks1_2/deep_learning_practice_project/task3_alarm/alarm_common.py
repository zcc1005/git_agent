from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT = (
    "你是工业皮带异物检测系统的报警报告生成助手。"
    "请根据检测JSON生成结构清晰、语言规范、适合现场处置的工业报警报告。"
)
ALARM_INSTRUCTION = (
    "请根据工业皮带异物检测JSON生成规范的报警报告，包含报警结论、目标信息、风险说明和处理建议。"
)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"没有找到检测结果文件：{path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"检测结果 JSON 解析失败：{path}，第 {exc.lineno} 行：{exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"检测结果必须是 JSON 对象：{path}")
    return data


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")


def _image_size_from_path(path_value: Any) -> Optional[Tuple[float, float]]:
    if not path_value:
        return None
    path = Path(str(path_value))
    if not path.exists():
        return None
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
        return float(width), float(height)
    except Exception:
        return None


def _fallback_canvas_size(objects: Iterable[Dict[str, Any]]) -> Tuple[float, float]:
    max_x = 1.0
    max_y = 1.0
    for obj in objects:
        bbox = obj.get("bbox_xyxy") or obj.get("bbox") or []
        if len(bbox) >= 4:
            max_x = max(max_x, float(bbox[2]))
            max_y = max(max_y, float(bbox[3]))
    return max_x, max_y


def estimate_position(
    bbox: List[float],
    image_width: Optional[float] = None,
    image_height: Optional[float] = None,
) -> str:
    if len(bbox) < 4:
        return "未知区域"

    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    width = image_width or max(x2, 1.0)
    height = image_height or max(y2, 1.0)
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0

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


def adapt_detection_for_alarm(
    detection: Dict[str, Any],
    top_k: int = 5,
    sort_by: str = "confidence",
) -> Dict[str, Any]:
    raw_objects = detection.get("objects") or []
    if not isinstance(raw_objects, list):
        raw_objects = []

    fallback_width, fallback_height = _fallback_canvas_size(
        [obj for obj in raw_objects if isinstance(obj, dict)]
    )

    valid_objects: List[Dict[str, Any]] = []
    for obj in raw_objects:
        if not isinstance(obj, dict):
            continue
        confidence = float(obj.get("confidence", 0.0) or 0.0)
        area = float(obj.get("area", 0.0) or 0.0)
        valid_objects.append({**obj, "_confidence": confidence, "_area": area})

    if sort_by == "area":
        valid_objects.sort(key=lambda item: item.get("_area", 0.0), reverse=True)
    else:
        valid_objects.sort(key=lambda item: item.get("_confidence", 0.0), reverse=True)

    selected_objects = valid_objects[:top_k]
    adapted_objects: List[Dict[str, Any]] = []

    for obj in selected_objects:
        bbox = obj.get("bbox") or obj.get("bbox_xyxy") or [0, 0, 0, 0]
        bbox = [round(float(v), 2) for v in bbox[:4]]
        image_path = obj.get("image") or obj.get("image_name") or ""
        image_size = _image_size_from_path(image_path)
        image_width, image_height = image_size or (fallback_width, fallback_height)

        raw_class = str(obj.get("class", "")).strip()
        if raw_class == "yiwu" or not raw_class:
            class_value = "unknown"
            class_name = "工业皮带异物"
        else:
            class_value = raw_class
            class_name = str(obj.get("class_name") or raw_class)

        adapted_objects.append(
            {
                "image_name": Path(str(image_path)).name if image_path else str(obj.get("image_name", "")),
                "class": class_value,
                "class_name": class_name,
                "confidence": round(float(obj.get("confidence", 0.0) or 0.0), 4),
                "bbox": bbox,
                "area": round(float(obj.get("area", 0.0) or 0.0), 2),
                "position": str(obj.get("position") or estimate_position(bbox, image_width, image_height)),
            }
        )

    representative_image = adapted_objects[0]["image_name"] if adapted_objects else ""
    return {
        "image_name": representative_image or "multiple_images",
        "objects": adapted_objects,
        "summary": {
            "status": detection.get("status", "unknown"),
            "num_images": int(detection.get("num_images", 0) or 0),
            "num_detections": int(detection.get("num_detections", len(raw_objects)) or 0),
            "has_yiwu": bool(detection.get("has_yiwu", len(raw_objects) > 0)),
            "selected_top_k": len(adapted_objects),
            "selection_rule": f"按{('面积' if sort_by == 'area' else '置信度')}排序保留前 {top_k} 个目标",
        },
    }


def build_alarm_prompt(detection: Dict[str, Any], adapted: Dict[str, Any]) -> str:
    summary = adapted.get("summary", {})
    detection_json = json.dumps(adapted, ensure_ascii=False, indent=2)
    has_yiwu_text = "是" if summary.get("has_yiwu") else "否"

    return (
        "请根据以下工业皮带异物检测结果生成规范报警报告。\n"
        "要求包含：\n"
        "1. 报警结论\n"
        "2. 风险等级\n"
        "3. 目标信息\n"
        "4. 风险说明\n"
        "5. 处理建议\n\n"
        "检测统计：\n"
        f"- 检测图片数量：{summary.get('num_images', detection.get('num_images', 0))}\n"
        f"- 检测目标数量：{summary.get('num_detections', detection.get('num_detections', 0))}\n"
        f"- 是否检测到异物：{has_yiwu_text}\n"
        f"- 送入模型分析的目标：{summary.get('selected_top_k', 0)} 个，"
        f"{summary.get('selection_rule', '按置信度排序保留前 5 个目标')}\n\n"
        "检测JSON：\n"
        f"{detection_json}"
    )


def ensure_report_sections(report: str) -> str:
    fallback_sections = {
        "报警结论": "检测到工业皮带异物，当前存在皮带输送安全风险。",
        "风险等级": "高风险",
        "目标信息": "本次检测结果显示存在异物目标，请结合 detection.json 和检测可视化图片复核目标位置、置信度和检测框信息。",
        "风险说明": "异物可能造成皮带划伤、滚筒卡滞、设备异常停机或物料输送堵塞。",
        "处理建议": "建议立即触发声光报警，降低或停止皮带运行，安排现场人员检查并清理异物，确认设备状态正常后恢复运行。",
    }
    text = report.strip()
    if not text:
        return text
    if "工业皮带异物报警报告" not in text[:80]:
        text = "工业皮带异物报警报告\n\n" + text

    missing = [item for item in fallback_sections if item not in text]
    if missing:
        lines = text.splitlines()
        title = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        supplements = []
        for section_name in missing:
            supplements.append(f"{section_name}：{fallback_sections[section_name]}")
        text = title + "\n" + "\n".join(supplements)
        if body:
            text += "\n" + body
    return text
