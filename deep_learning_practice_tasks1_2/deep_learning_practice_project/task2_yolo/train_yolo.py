from __future__ import annotations

# Allow running this file directly from PyCharm or the command line.
import sys
from pathlib import Path as _PathForSys

sys.path.append(str(_PathForSys(__file__).resolve().parents[1]))

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

from task2_yolo.check_yolo_dataset import audit_dataset, print_report
from project_config import YOLO_RUN_NAME
from task2_yolo.yolo_config import RUN_DIR, YOLO_DATA_DIR


def _fmt_metric(value: object) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def print_training_summary(results: object, save_dir: Path) -> None:
    results_dict = getattr(results, "results_dict", {}) or {}
    names = getattr(results, "names", {}) or {}
    maps = list(getattr(results, "maps", []) or [])
    nt_per_class = list(getattr(results, "nt_per_class", []) or [])

    print("\nTraining summary")
    print(f"  save_dir: {save_dir}")
    print(f"  best.pt:  {save_dir / 'weights' / 'best.pt'}")
    print(f"  last.pt:  {save_dir / 'weights' / 'last.pt'}")
    print("  overall:")
    print(f"    precision: {_fmt_metric(results_dict.get('metrics/precision(B)'))}")
    print(f"    recall:    {_fmt_metric(results_dict.get('metrics/recall(B)'))}")
    print(f"    mAP50:     {_fmt_metric(results_dict.get('metrics/mAP50(B)'))}")
    print(f"    mAP50-95:  {_fmt_metric(results_dict.get('metrics/mAP50-95(B)'))}")

    results_csv = save_dir / "results.csv"
    if results_csv.exists():
        with results_csv.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            key = "metrics/mAP50-95(B)"
            best_row = max(rows, key=lambda row: float(row.get(key) or 0.0))
            last_row = rows[-1]
            print("  training curve:")
            print(
                f"    best epoch: {best_row.get('epoch')} "
                f"mAP50-95={_fmt_metric(best_row.get(key))} "
                f"mAP50={_fmt_metric(best_row.get('metrics/mAP50(B)'))} "
                f"recall={_fmt_metric(best_row.get('metrics/recall(B)'))}"
            )
            print(
                f"    last epoch: {last_row.get('epoch')} "
                f"mAP50-95={_fmt_metric(last_row.get(key))} "
                f"mAP50={_fmt_metric(last_row.get('metrics/mAP50(B)'))} "
                f"recall={_fmt_metric(last_row.get('metrics/recall(B)'))}"
            )

    if maps:
        print("  per-class mAP50-95:")
        for class_id, class_map in enumerate(maps):
            class_name = names.get(class_id, str(class_id))
            instances = nt_per_class[class_id] if class_id < len(nt_per_class) else "n/a"
            print(f"    {class_id}:{class_name:<8} instances={instances:<4} mAP50-95={_fmt_metric(class_map)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO for typed foreign objects")
    parser.add_argument("--data", type=Path, default=YOLO_DATA_DIR / "data.yaml")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="Model checkpoint, for example yolov8n.pt or yolov8s.pt.",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default=None, help="Example: 0 or cpu.")
    parser.add_argument("--project", type=Path, default=RUN_DIR)
    parser.add_argument("--name", type=str, default=YOLO_RUN_NAME)
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Reuse the same run directory. By default, Ultralytics creates a new numbered directory.",
    )
    parser.add_argument(
        "--allow-data-issues",
        action="store_true",
        help="Train even if the dataset audit reports errors. Not recommended.",
    )
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"data.yaml not found: {args.data}\n"
            "Run: python task2_yolo/prepare_yolo_dataset.py"
        )

    audit_report = audit_dataset(args.data)
    print_report(audit_report)
    if not audit_report.ok and not args.allow_data_issues:
        raise RuntimeError(
            "Dataset audit failed. Fix the ERROR lines above before training, "
            "or pass --allow-data-issues if you understand the risk."
        )

    model = YOLO(args.model)
    train_kwargs = dict(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name=args.name,
        exist_ok=args.exist_ok,
        workers=0,
    )
    if args.device is not None:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)
    save_dir = getattr(results, "save_dir", args.project / args.name)
    print_training_summary(results, Path(save_dir))


if __name__ == "__main__":
    main()
