from __future__ import annotations

# 允许直接用 PyCharm/命令行运行本文件，而不必使用 python -m。
import sys
from pathlib import Path as _PathForSys
sys.path.append(str(_PathForSys(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from ultralytics import YOLO

from task2_yolo.yolo_config import RUN_DIR, YOLO_DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="Train YOLO for typed foreign objects")
    parser.add_argument("--data", type=Path, default=YOLO_DATA_DIR / "data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="可改为 yolov8s.pt；无网络时可先手动下载权重")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default=None, help="例如 0 或 cpu；默认自动选择")
    parser.add_argument("--project", type=Path, default=RUN_DIR)
    parser.add_argument("--name", type=str, default="foreign_objects_yolov8n")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"找不到 data.yaml：{args.data}\n请先运行：python task2_yolo/prepare_yolo_dataset.py"
        )

    model = YOLO(args.model)
    train_kwargs = dict(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        workers=0,
    )
    if args.device is not None:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)
    print("训练完成。")
    print(f"最佳权重通常位于：{args.project / args.name / 'weights' / 'best.pt'}")
    print(results)


if __name__ == "__main__":
    main()
