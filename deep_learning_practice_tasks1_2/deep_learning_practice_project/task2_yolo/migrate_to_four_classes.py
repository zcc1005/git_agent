from __future__ import annotations

"""One-time migration from five YOLO classes to four known classes.

Old IDs: 0 unknown, 1 stone, 2 plastic, 3 metal, 4 wood.
New IDs: 0 stone, 1 plastic, 2 metal, 3 wood.

Any image containing an old class-0 box is moved with its unchanged label to
data/yolo_unknown_eval. It is deliberately excluded from normal YOLO training.
"""

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_config import DATA_DIR, YOLO_DATA_DIR


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OLD_NAMES = ["unknown", "stone", "plastic", "metal", "wood"]
NEW_NAMES = ["stone", "plastic", "metal", "wood"]


@dataclass(frozen=True)
class LabelPlan:
    split: str
    image_path: Path
    label_path: Path
    lines: tuple[str, ...]
    contains_unknown: bool
    contains_known: bool


def read_names(data_yaml: Path) -> list[str]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    raw_names = data.get("names", {})
    if isinstance(raw_names, list):
        return [str(name) for name in raw_names]
    if isinstance(raw_names, dict):
        return [str(raw_names[key]) for key in sorted(raw_names, key=int)]
    raise ValueError(f"Invalid names in {data_yaml}")


def find_image(image_dir: Path, stem: str) -> Path:
    matches = [
        path
        for path in image_dir.glob(f"{stem}.*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one image for label {stem}.txt in {image_dir}, "
            f"found {len(matches)}"
        )
    return matches[0]


def build_plan(data_dir: Path, unknown_dir: Path) -> list[LabelPlan]:
    plans: list[LabelPlan] = []
    for split in ("train", "val", "test"):
        image_dir = data_dir / "images" / split
        label_dir = data_dir / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing images/labels directory for split: {split}")

        for label_path in sorted(label_dir.glob("*.txt")):
            image_path = find_image(image_dir, label_path.stem)
            source_lines = tuple(
                line.strip()
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            class_ids: list[int] = []
            for line_number, line in enumerate(source_lines, 1):
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f"{label_path}:{line_number} must have 5 columns")
                try:
                    class_id = int(parts[0])
                    [float(value) for value in parts[1:]]
                except ValueError as exc:
                    raise ValueError(f"{label_path}:{line_number} contains non-numeric data") from exc
                if class_id not in range(5):
                    raise ValueError(f"{label_path}:{line_number} has invalid old class ID {class_id}")
                class_ids.append(class_id)

            contains_unknown = 0 in class_ids
            if contains_unknown:
                image_target = unknown_dir / "images" / split / image_path.name
                label_target = unknown_dir / "labels" / split / label_path.name
                if image_target.exists() or label_target.exists():
                    raise FileExistsError(f"Unknown-eval destination already exists for {image_path.name}")

            plans.append(
                LabelPlan(
                    split=split,
                    image_path=image_path,
                    label_path=label_path,
                    lines=source_lines,
                    contains_unknown=contains_unknown,
                    contains_known=any(class_id > 0 for class_id in class_ids),
                )
            )
    return plans


def remap_known_lines(lines: tuple[str, ...]) -> str:
    remapped: list[str] = []
    for line in lines:
        parts = line.split()
        parts[0] = str(int(parts[0]) - 1)
        remapped.append(" ".join(parts))
    return "\n".join(remapped) + ("\n" if remapped else "")


def write_yaml(path: Path, names: list[str]) -> None:
    data = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": {class_id: name for class_id, name in enumerate(names)},
    }
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def migrate(data_dir: Path, unknown_dir: Path, apply: bool) -> None:
    data_yaml = data_dir / "data.yaml"
    current_names = read_names(data_yaml)
    if current_names == NEW_NAMES:
        print("Dataset is already using the four-class mapping; nothing to do.")
        return
    if current_names != OLD_NAMES:
        raise ValueError(f"Expected old classes {OLD_NAMES}, found {current_names}")

    plans = build_plan(data_dir, unknown_dir)
    unknown = [plan for plan in plans if plan.contains_unknown]
    mixed = [plan for plan in unknown if plan.contains_known]
    print(f"Labels checked: {len(plans)}")
    print(f"Images to move to unknown_eval: {len(unknown)} (mixed: {len(mixed)})")
    print(f"Known/background labels to keep and remap: {len(plans) - len(unknown)}")

    if not apply:
        print("Dry run only. Re-run with --apply to perform the migration.")
        return

    for plan in plans:
        if plan.contains_unknown:
            image_target = unknown_dir / "images" / plan.split / plan.image_path.name
            label_target = unknown_dir / "labels" / plan.split / plan.label_path.name
            image_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(plan.image_path), str(image_target))
            shutil.move(str(plan.label_path), str(label_target))
        else:
            plan.label_path.write_text(remap_known_lines(plan.lines), encoding="utf-8")

    write_yaml(data_yaml, NEW_NAMES)
    unknown_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(unknown_dir / "data.yaml", OLD_NAMES)
    (unknown_dir / "README.md").write_text(
        "# Unknown 拒识评估数据\n\n"
        "这些图片因为包含原类别 0（unknown）而从四分类训练集中移出。"
        "标签继续保留原五分类 ID，因此没有丢失任何标注信息。"
        "该目录只用于调整 0.15/0.40 双阈值，不要作为四分类 YOLO 训练集。\n",
        encoding="utf-8",
    )
    print("Migration completed.")
    print(f"Four-class dataset: {data_dir}")
    print(f"Unknown evaluation archive: {unknown_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate YOLO labels from five to four classes")
    parser.add_argument("--data-dir", type=Path, default=YOLO_DATA_DIR)
    parser.add_argument("--unknown-dir", type=Path, default=DATA_DIR / "yolo_unknown_eval")
    parser.add_argument("--apply", action="store_true", help="Perform changes; default is dry-run")
    args = parser.parse_args()
    migrate(args.data_dir.resolve(), args.unknown_dir.resolve(), args.apply)


if __name__ == "__main__":
    main()
