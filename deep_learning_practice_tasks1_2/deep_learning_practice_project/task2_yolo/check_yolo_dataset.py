from __future__ import annotations

"""Validate a YOLO detection dataset before starting a costly training run."""

import argparse
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Issue:
    level: str
    message: str


@dataclass
class SplitStats:
    name: str
    image_count: int = 0
    labeled_image_count: int = 0
    background_count: int = 0
    box_count: int = 0
    class_counts: Counter[int] = field(default_factory=Counter)
    missing_labels: list[Path] = field(default_factory=list)
    orphan_labels: list[Path] = field(default_factory=list)


@dataclass
class AuditReport:
    data_yaml: Path
    class_names: dict[int, str]
    splits: dict[str, SplitStats] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(issue.level == "ERROR" for issue in self.issues)


def _normalise_names(raw_names: object) -> dict[int, str]:
    if isinstance(raw_names, list):
        names = {i: str(name) for i, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        names = {int(class_id): str(name) for class_id, name in raw_names.items()}
    else:
        raise ValueError("data.yaml 的 names 必须是列表或字典")

    expected_ids = list(range(len(names)))
    if sorted(names) != expected_ids:
        raise ValueError(f"类别 ID 必须从 0 连续编号，当前为 {sorted(names)}")
    return names


def _replace_images_with_labels(image_dir: Path, dataset_root: Path, split: str) -> Path:
    parts = list(image_dir.parts)
    image_positions = [i for i, part in enumerate(parts) if part.lower() == "images"]
    if image_positions:
        parts[image_positions[-1]] = "labels"
        return Path(*parts)
    return dataset_root / "labels" / split


def _iter_nonempty_lines(path: Path) -> Iterable[tuple[int, str]]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if line:
            yield line_number, line


def audit_dataset(data_yaml: Path) -> AuditReport:
    data_yaml = data_yaml.resolve()
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("data.yaml 内容为空或格式不正确")

    class_names = _normalise_names(data.get("names"))
    declared_nc = data.get("nc")
    if declared_nc is not None and int(declared_nc) != len(class_names):
        raise ValueError(
            f"data.yaml 的 nc={declared_nc}，但 names 中有 {len(class_names)} 类"
        )

    raw_root = data.get("path")
    if raw_root:
        dataset_root = Path(raw_root)
        if not dataset_root.is_absolute():
            dataset_root = data_yaml.parent / dataset_root
    else:
        dataset_root = data_yaml.parent
    dataset_root = dataset_root.resolve()

    report = AuditReport(data_yaml=data_yaml, class_names=class_names)

    for split in ("train", "val", "test"):
        raw_image_dir = data.get(split)
        if not raw_image_dir:
            if split in {"train", "val"}:
                report.issues.append(Issue("ERROR", f"data.yaml 缺少 {split} 路径"))
            continue
        if isinstance(raw_image_dir, list):
            report.issues.append(
                Issue("ERROR", f"当前检查器暂不支持 {split} 的多路径列表")
            )
            continue

        image_dir = Path(raw_image_dir)
        if not image_dir.is_absolute():
            image_dir = dataset_root / image_dir
        image_dir = image_dir.resolve()
        label_dir = _replace_images_with_labels(image_dir, dataset_root, split)
        stats = SplitStats(name=split)
        report.splits[split] = stats

        if not image_dir.is_dir():
            report.issues.append(Issue("ERROR", f"{split} 图片目录不存在：{image_dir}"))
            continue
        if not label_dir.is_dir():
            report.issues.append(Issue("ERROR", f"{split} 标签目录不存在：{label_dir}"))
            continue

        images = sorted(
            path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS
        )
        labels = sorted(label_dir.glob("*.txt"))
        image_stems = {path.stem for path in images}
        stats.image_count = len(images)
        stats.orphan_labels = [path for path in labels if path.stem not in image_stems]

        if stats.orphan_labels:
            examples = ", ".join(path.name for path in stats.orphan_labels[:3])
            report.issues.append(
                Issue("WARN", f"{split} 有 {len(stats.orphan_labels)} 个孤立标签：{examples}")
            )

        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                stats.missing_labels.append(label_path)
                continue

            lines = list(_iter_nonempty_lines(label_path))
            if not lines:
                stats.background_count += 1
                continue

            stats.labeled_image_count += 1
            seen_lines: set[str] = set()
            for line_number, line in lines:
                parts = line.split()
                location = f"{label_path.name}:{line_number}"
                if len(parts) != 5:
                    report.issues.append(
                        Issue("ERROR", f"{split}/{location} 应有 5 列，实际为 {len(parts)} 列")
                    )
                    continue

                try:
                    class_id = int(parts[0])
                    coords = [float(value) for value in parts[1:]]
                except ValueError:
                    report.issues.append(Issue("ERROR", f"{split}/{location} 含非数字字段"))
                    continue

                if class_id not in class_names:
                    report.issues.append(
                        Issue("ERROR", f"{split}/{location} 类别 ID {class_id} 不在 names 中")
                    )
                    continue
                if not all(math.isfinite(value) for value in coords):
                    report.issues.append(Issue("ERROR", f"{split}/{location} 坐标含 NaN/Inf"))
                    continue

                x_center, y_center, width, height = coords
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                    report.issues.append(Issue("ERROR", f"{split}/{location} 中心坐标不在 [0, 1]"))
                    continue
                if not (0 < width <= 1 and 0 < height <= 1):
                    report.issues.append(Issue("ERROR", f"{split}/{location} 宽高不在 (0, 1]"))
                    continue

                normalised_line = " ".join(parts)
                if normalised_line in seen_lines:
                    report.issues.append(Issue("WARN", f"{split}/{location} 是重复框"))
                seen_lines.add(normalised_line)
                stats.box_count += 1
                stats.class_counts[class_id] += 1

        if stats.missing_labels:
            examples = ", ".join(path.name for path in stats.missing_labels[:4])
            level = "ERROR" if split in {"train", "val"} else "WARN"
            report.issues.append(
                Issue(level, f"{split} 缺少 {len(stats.missing_labels)} 个标签文件：{examples}")
            )

        if split in {"train", "val"}:
            absent = [
                f"{class_id}:{class_names[class_id]}"
                for class_id in class_names
                if stats.class_counts[class_id] == 0
            ]
            if absent:
                report.issues.append(
                    Issue("ERROR", f"{split} 完全缺少这些类别的实例：{', '.join(absent)}")
                )

        nonzero_counts = [count for count in stats.class_counts.values() if count > 0]
        if split == "train" and len(nonzero_counts) >= 2:
            imbalance = max(nonzero_counts) / min(nonzero_counts)
            if imbalance >= 10:
                report.issues.append(
                    Issue("WARN", f"train 最大/最小类别框数相差 {imbalance:.1f} 倍，类别严重不均衡")
                )

        if split == "test" and stats.box_count == 0:
            report.issues.append(
                Issue("WARN", "test 没有任何标注框，只能用于推理展示，不能计算测试集指标")
            )

    return report


def print_report(report: AuditReport) -> None:
    print(f"Dataset YAML: {report.data_yaml}")
    print("Classes: " + ", ".join(f"{i}:{name}" for i, name in report.class_names.items()))
    for split, stats in report.splits.items():
        counts = ", ".join(
            f"{class_id}:{stats.class_counts[class_id]}" for class_id in report.class_names
        )
        print(
            f"{split:>5}: images={stats.image_count}, labeled={stats.labeled_image_count}, "
            f"backgrounds={stats.background_count}, boxes={stats.box_count}, classes=[{counts}]"
        )

    if report.issues:
        print("Issues:")
        for issue in report.issues:
            print(f"  [{issue.level}] {issue.message}")
    else:
        print("Issues: none")
    print("Result: PASS" if report.ok else "Result: FAIL")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check YOLO paths, labels and class distribution")
    parser.add_argument("--data", type=Path, required=True, help="Path to data.yaml")
    args = parser.parse_args()
    report = audit_dataset(args.data)
    print_report(report)
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
