from __future__ import annotations

# 允许直接用 PyCharm/命令行运行本文件，而不必使用 python -m。
import sys
from pathlib import Path as _PathForSys
sys.path.append(str(_PathForSys(__file__).resolve().parents[1]))

import argparse
import shutil
import zipfile
from pathlib import Path

import yaml

from task2_yolo.yolo_config import CLASS_NAMES, RAW_ZIP, YOLO_DATA_DIR

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_split_dir(root: Path, split: str) -> Path | None:
    candidates = []
    aliases = [split]
    if split == "val":
        aliases += ["valid", "validation"]
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() in aliases:
            imgs = [x for x in p.iterdir() if x.suffix.lower() in IMAGE_EXTS]
            if imgs:
                candidates.append(p)
    return candidates[0] if candidates else None


def copy_images_and_make_empty_labels(src_dir: Path, dst_root: Path, split: str) -> int:
    img_dst = dst_root / "images" / split
    label_dst = dst_root / "labels" / split
    img_dst.mkdir(parents=True, exist_ok=True)
    label_dst.mkdir(parents=True, exist_ok=True)

    count = 0
    for img in sorted(src_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        new_img = img_dst / img.name
        if not new_img.exists():
            shutil.copy2(img, new_img)
        # 先创建空标签。后续 annotate_yiwu.py 会写入带类别的异物框。
        label_file = label_dst / (img.stem + ".txt")
        if not label_file.exists():
            label_file.write_text("", encoding="utf-8")
        count += 1
    return count


def write_data_yaml(dst_root: Path):
    data = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    with (dst_root / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description="Extract uploaded belt images and convert to YOLO folder structure")
    parser.add_argument("--zip", type=Path, default=RAW_ZIP)
    parser.add_argument("--out", type=Path, default=YOLO_DATA_DIR)
    parser.add_argument("--force", action="store_true", help="重新生成 data/yolo_yiwu")
    args = parser.parse_args()

    if not args.zip.exists():
        raise FileNotFoundError(f"找不到图片压缩包：{args.zip}")

    if args.force and args.out.exists():
        shutil.rmtree(args.out)

    extract_dir = args.out.parent / "yolo_raw_extracted"
    if args.force and extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"正在解压：{args.zip}")
    with zipfile.ZipFile(args.zip, "r") as zf:
        zf.extractall(extract_dir)

    split_map = {
        "train": find_split_dir(extract_dir, "train"),
        "val": find_split_dir(extract_dir, "val"),
        "test": find_split_dir(extract_dir, "test"),
    }

    if split_map["train"] is None:
        raise RuntimeError("压缩包中没有找到 train 图片目录。")
    if split_map["val"] is None:
        print("没有找到 valid/val 目录，将从 train 中继续使用空 val 目录，建议自行补充验证集。")
    if split_map["test"] is None:
        print("没有找到 test 目录，将创建空 test 目录。")

    args.out.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        src = split_map.get(split)
        if src is None:
            continue
        n = copy_images_and_make_empty_labels(src, args.out, split)
        print(f"{split}: 复制 {n} 张图片")

    write_data_yaml(args.out)
    print(f"YOLO 数据集已生成：{args.out}")
    print("类别：")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i} -> {name}")
    print("下一步：python task2_yolo/annotate_yiwu.py --split train")


if __name__ == "__main__":
    main()
