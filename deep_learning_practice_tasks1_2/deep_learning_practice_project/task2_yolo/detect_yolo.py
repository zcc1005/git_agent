from __future__ import annotations

# 允许直接用 PyCharm/命令行运行本文件，而不必使用 python -m。
import sys
from pathlib import Path as _PathForSys

PROJECT_ROOT = _PathForSys(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import cv2
from ultralytics import YOLO

from project_config import OUTPUTS_DIR, YOLO_MODEL_PATH
from task2_yolo.yolo_config import CLASS_DISPLAY_NAMES, CLASS_ID_TO_NAME, CLASS_NAMES


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIS_COLORS = {
    "unknown": (160, 160, 160),
    "stone": (64, 64, 255),
    "plastic": (0, 200, 255),
    "metal": (255, 128, 0),
    "wood": (0, 180, 80),
}


def normalize_class_name(class_id: int, names: Dict[int, str] | None = None) -> tuple[str, str]:
    if names and class_id in names:
        raw_name = str(names[class_id]).strip()
    elif class_id in CLASS_ID_TO_NAME:
        raw_name = CLASS_ID_TO_NAME[class_id]
    else:
        raw_name = str(class_id)

    lowered = raw_name.lower()
    if raw_name in CLASS_DISPLAY_NAMES.values():
        class_key = next(
            (key for key, value in CLASS_DISPLAY_NAMES.items() if value == raw_name),
            lowered,
        )
        return class_key, raw_name

    if lowered in {"yiwu", "foreign_object", "foreign-object", "foreign object"}:
        lowered = "unknown"

    display_name = CLASS_DISPLAY_NAMES.get(lowered, raw_name)
    return lowered, display_name


def validate_four_class_model(model: YOLO) -> None:
    raw_names = getattr(model, "names", {})
    if isinstance(raw_names, dict):
        model_names = [str(raw_names[key]) for key in sorted(raw_names)]
    else:
        model_names = [str(name) for name in raw_names]
    if model_names != CLASS_NAMES:
        raise RuntimeError(
            "当前权重不是新的四分类模型。"
            f"模型类别={model_names}，期望类别={CLASS_NAMES}。"
            "请先重新训练，并使用 yiwu_yolov8s_4class/weights/best.pt。"
        )


def save_visualization(result: Any, image_objects: List[Dict[str, Any]], vis_dir: Path, index: int) -> Path:
    image = result.orig_img.copy()
    for obj in image_objects:
        x1, y1, x2, y2 = (int(round(value)) for value in obj["bbox_xyxy"])
        class_key = str(obj["class"])
        color = VIS_COLORS.get(class_key, (0, 255, 0))
        label = f"{class_key} {obj['confidence']:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text_y = max(20, y1 - 7)
        cv2.putText(image, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    source_path = Path(str(result.path))
    if source_path.suffix.lower() in IMAGE_EXTS:
        output_path = vis_dir / source_path.name
    else:
        output_path = vis_dir / f"{source_path.stem}_{index:06d}.jpg"
    suffix = output_path.suffix if output_path.suffix.lower() in IMAGE_EXTS else ".jpg"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise RuntimeError(f"无法编码检测可视化图片：{output_path}")
    encoded.tofile(str(output_path))
    return output_path


def read_command(command_path: Path) -> Dict[str, Any]:
    """
    读取任务一生成的 command.json。
    如果文件不存在，返回空字典。
    """
    if not command_path.exists():
        return {}

    try:
        with command_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"读取 command.json 失败：{command_path}，原因：{e}")
        return {}


def should_start_detection(command: Dict[str, Any]) -> bool:
    """
    判断是否应该启动检测。
    command 为 go 或 start_detection 为 true 时启动。
    """
    if not command:
        print("未找到 command.json 或 command.json 为空，默认允许检测。")
        return True

    cmd = str(command.get("command", "")).lower().strip()
    start_detection = bool(command.get("start_detection", False))

    return cmd == "go" or start_detection is True


def make_skipped_json(
    output_json: Path,
    source: Path,
    model_path: Path,
    command: Dict[str, Any],
) -> None:
    """
    当 command 不是 go 时，跳过检测，但仍然生成 detection.json。
    """
    result = {
        "status": "skipped",
        "reason": "command is not go and start_detection is not true",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(source),
        "model_path": str(model_path),
        "command": command,
        "num_images": 0,
        "num_detections": 0,
        "has_yiwu": False,
        "has_foreign_object": False,
        "class_counts": {},
        "objects": [],
        "visualization_dir": None,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"command 不是 go，已跳过检测。detection.json 已保存：{output_json}")


def detect_yiwu(
    source: Path,
    model_path: Path,
    output_json: Path,
    conf: float = 0.15,
    known_conf: float = 0.40,
) -> None:
    """
    加载 YOLO best.pt，对 source 中的图片进行异物检测。
    同时保存：
    1. outputs/detection.json
    2. outputs/detections_vis/ 带框检测图片
    """
    output_dir = output_json.parent
    vis_dir = output_dir / "detections_vis"

    if not 0.0 <= conf < known_conf <= 1.0:
        raise ValueError(
            f"阈值必须满足 0 <= conf < known_conf <= 1，当前为 {conf}, {known_conf}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(
            f"没有找到 YOLO 模型权重：{model_path}\n"
            "请确认四分类模型已训练完成，默认路径为 "
            "runs/yolo/yiwu_yolov8s_4class/weights/best.pt"
        )

    if not source.exists():
        raise FileNotFoundError(f"没有找到待检测图片或文件夹：{source}")

    print(f"加载模型：{model_path}")
    print(f"检测输入：{source}")
    print(f"双阈值：<{conf:.2f} 忽略，{conf:.2f}-{known_conf:.2f} unknown，>={known_conf:.2f} 已知类别")

    model = YOLO(str(model_path))
    validate_four_class_model(model)

    # 使用低阈值保留候选框，再在后处理中执行 unknown 拒识。
    results = model.predict(
        source=str(source),
        conf=conf,
        imgsz=640,
        save=False,
        verbose=False,
    )

    objects: List[Dict[str, Any]] = []
    class_counts: Dict[str, int] = {}
    num_images = 0
    num_detections = 0

    for result_index, result in enumerate(results, 1):
        num_images += 1

        image_path = str(result.path)
        boxes = result.boxes
        result_names = getattr(result, "names", None)

        image_objects: List[Dict[str, Any]] = []
        if boxes is not None:
            for box in boxes:
                predicted_class_id = int(box.cls[0].item())
                predicted_key, predicted_name = normalize_class_name(
                    predicted_class_id, result_names
                )
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                rejected_as_unknown = confidence < known_conf
                if rejected_as_unknown:
                    class_id = None
                    class_key = "unknown"
                    class_name = CLASS_DISPLAY_NAMES["unknown"]
                else:
                    class_id = predicted_class_id
                    class_key = predicted_key
                    class_name = predicted_name

                area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

                obj = {
                    "image": image_path,
                    "class_id": class_id,
                    "class": class_key,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "rejected_as_unknown": rejected_as_unknown,
                    "predicted_class_id": predicted_class_id,
                    "predicted_class": predicted_key,
                    "predicted_class_name": predicted_name,
                    "bbox_xyxy": [
                        round(float(x1), 2),
                        round(float(y1), 2),
                        round(float(x2), 2),
                        round(float(y2), 2),
                    ],
                    "area": round(area, 2),
                }

                objects.append(obj)
                image_objects.append(obj)
                num_detections += 1

        save_visualization(result, image_objects, vis_dir, result_index)

    detection_result = {
        "status": "detected",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(source),
        "model_path": str(model_path),
        "thresholds": {
            "detection_min_confidence": conf,
            "known_class_min_confidence": known_conf,
        },
        "num_images": num_images,
        "num_detections": num_detections,
        "has_yiwu": num_detections > 0,
        "has_foreign_object": num_detections > 0,
        "class_names": CLASS_NAMES,
        "class_display_names": CLASS_DISPLAY_NAMES,
        "class_counts": class_counts,
        "objects": objects,
        "visualization_dir": str(vis_dir),
    }

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(detection_result, f, ensure_ascii=False, indent=2)

    print(f"detection.json 已保存：{output_json}")
    print(f"检测可视化图片已保存：{vis_dir}")
    print(
        {
            "num_images": num_images,
            "num_detections": num_detections,
            "has_yiwu": num_detections > 0,
            "class_counts": class_counts,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO typed foreign object detection script")

    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "data" / "yolo_yiwu" / "images" / "test",
        help="待检测图片、文件夹或视频路径",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=YOLO_MODEL_PATH,
        help="YOLO 模型权重 best.pt 路径",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS_DIR / "detection.json",
        help="检测结果 JSON 输出路径",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="最低检测阈值；低于该值作为背景忽略（默认 0.15）",
    )

    parser.add_argument(
        "--known-conf",
        type=float,
        default=0.40,
        help="已知类别阈值；conf 到该值之间输出 unknown（默认 0.40）",
    )

    parser.add_argument(
        "--ignore_command",
        action="store_true",
        help="忽略 outputs/command.json，直接执行检测",
    )

    args = parser.parse_args()

    command_path = PROJECT_ROOT / "outputs" / "command.json"
    command = read_command(command_path)

    if not args.ignore_command:
        if not should_start_detection(command):
            make_skipped_json(
                output_json=args.output,
                source=args.source,
                model_path=args.model,
                command=command,
            )
            return
    else:
        print("已启用 --ignore_command，跳过 command.json 检查，直接执行检测。")

    detect_yiwu(
        source=args.source,
        model_path=args.model,
        output_json=args.output,
        conf=args.conf,
        known_conf=args.known_conf,
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(e)
        raise SystemExit(1)
