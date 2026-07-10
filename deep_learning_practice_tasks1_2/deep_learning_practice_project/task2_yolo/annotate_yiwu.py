from __future__ import annotations

# Allow running this file directly from PyCharm or the command line.
import sys
from pathlib import Path as _PathForSys

PROJECT_ROOT = _PathForSys(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from task2_yolo.yolo_config import (
    CLASS_DISPLAY_NAMES,
    CLASS_ID_TO_NAME,
    CLASS_NAME_TO_ID,
    CLASS_NAMES,
    YOLO_DATA_DIR,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
Box = Tuple[int, int, int, int, int]
CLASS_COLORS = {
    CLASS_NAME_TO_ID["unknown"]: (180, 180, 180),
    CLASS_NAME_TO_ID["stone"]: (64, 64, 255),
    CLASS_NAME_TO_ID["plastic"]: (0, 200, 255),
    CLASS_NAME_TO_ID["metal"]: (255, 128, 0),
    CLASS_NAME_TO_ID["wood"]: (0, 180, 80),
}
KEY_TO_CLASS_ID = {
    ord("1"): CLASS_NAME_TO_ID["stone"],
    ord("2"): CLASS_NAME_TO_ID["plastic"],
    ord("3"): CLASS_NAME_TO_ID["metal"],
    ord("4"): CLASS_NAME_TO_ID["wood"],
    ord("5"): CLASS_NAME_TO_ID["unknown"],
}


def imread_unicode(path: Path):
    """Read images from paths that may contain Chinese characters on Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def class_label(class_id: int) -> str:
    name = CLASS_ID_TO_NAME.get(class_id, "unknown")
    display = CLASS_DISPLAY_NAMES.get(name, name)
    return f"{class_id}:{display}"


class YoloAnnotator:
    def __init__(
        self,
        data_dir: Path,
        split: str,
        start: int = 0,
        max_window_w: int = 1280,
        max_window_h: int = 800,
    ):
        self.data_dir = data_dir
        self.split = split

        self.image_dir = data_dir / "images" / split
        self.label_dir = data_dir / "labels" / split
        self.label_dir.mkdir(parents=True, exist_ok=True)

        if not self.image_dir.exists():
            raise RuntimeError(f"Image directory does not exist: {self.image_dir}")

        self.images = sorted(
            [p for p in self.image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        )

        if not self.images:
            raise RuntimeError(f"No images found: {self.image_dir}")

        self.idx = max(0, min(start, len(self.images) - 1))
        self.max_window_w = max_window_w
        self.max_window_h = max_window_h

        self.current_class_id = CLASS_NAME_TO_ID["stone"]
        self.boxes: List[Box] = []
        self.drawing = False
        self.start_xy = (0, 0)
        self.current_xy = (0, 0)

        self.scale = 1.0
        self.image = None
        self.win_name = (
            "Annotate foreign objects: 1 stone | 2 plastic | 3 metal | "
            "4 wood | 5 unknown | drag | s save | n none | u undo | r reset | q quit"
        )

        print(f"Dataset: {self.data_dir}")
        print(f"Current split: {self.split}")
        print(f"Images: {self.image_dir}")
        print(f"Labels: {self.label_dir}")
        print(f"Found {len(self.images)} images")
        print("Classes:")
        for class_id, name in enumerate(CLASS_NAMES):
            print(f"  {class_id}: {CLASS_DISPLAY_NAMES.get(name, name)}")
        print("Controls:")
        print("  1-5: select object type")
        print("  Mouse drag: draw a box using the selected type")
        print("  s: save current image and go next")
        print("  n: save empty label for no foreign object")
        print("  u: undo last box")
        print("  r: clear all boxes in current image")
        print("  q: quit")

    def yolo_label_path(self, image_path: Path) -> Path:
        return self.label_dir / f"{image_path.stem}.txt"

    def load_existing_boxes(self, image_path: Path, w: int, h: int) -> List[Box]:
        label_path = self.yolo_label_path(image_path)
        boxes: List[Box] = []

        if not label_path.exists():
            return boxes

        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            return boxes

        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls, xc, yc, bw, bh = map(float, parts)
            class_id = int(cls)
            if class_id not in CLASS_ID_TO_NAME:
                class_id = CLASS_NAMES.index("unknown")

            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            boxes.append((class_id, x1, y1, x2, y2))

        return boxes

    def save_boxes(self, image_path: Path, w: int, h: int):
        label_path = self.yolo_label_path(image_path)
        lines = []

        for class_id, x1, y1, x2, y2 in self.boxes:
            x1, x2 = sorted(
                [
                    max(0, min(w - 1, x1)),
                    max(0, min(w - 1, x2)),
                ]
            )
            y1, y2 = sorted(
                [
                    max(0, min(h - 1, y1)),
                    max(0, min(h - 1, y2)),
                ]
            )

            if x2 - x1 < 3 or y2 - y1 < 3:
                continue

            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"Saved: {label_path}, boxes={len(lines)}")

    def image_to_display(self, img):
        h, w = img.shape[:2]
        self.scale = min(self.max_window_w / w, self.max_window_h / h, 1.0)

        if self.scale < 1.0:
            return cv2.resize(img, (int(w * self.scale), int(h * self.scale)))

        return img.copy()

    def to_original_xy(self, x: int, y: int) -> Tuple[int, int]:
        return int(x / self.scale), int(y / self.scale)

    def draw_overlay(self):
        disp = self.image_to_display(self.image)

        for class_id, x1, y1, x2, y2 in self.boxes:
            color = CLASS_COLORS.get(class_id, (0, 255, 0))
            cv2.rectangle(
                disp,
                (int(x1 * self.scale), int(y1 * self.scale)),
                (int(x2 * self.scale), int(y2 * self.scale)),
                color,
                2,
            )
            cv2.putText(
                disp,
                class_label(class_id),
                (int(x1 * self.scale), max(20, int(y1 * self.scale) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        if self.drawing:
            x1, y1 = self.start_xy
            x2, y2 = self.current_xy
            color = CLASS_COLORS.get(self.current_class_id, (255, 0, 0))
            cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)

        text = (
            f"{self.idx + 1}/{len(self.images)}  {self.images[self.idx].name}  "
            f"boxes={len(self.boxes)}  current={class_label(self.current_class_id)}"
        )
        cv2.putText(
            disp,
            text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        return disp

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_xy = (x, y)
            self.current_xy = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_xy = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False

            x1d, y1d = self.start_xy
            x2d, y2d = x, y

            x1, y1 = self.to_original_xy(x1d, y1d)
            x2, y2 = self.to_original_xy(x2d, y2d)

            if abs(x2 - x1) >= 3 and abs(y2 - y1) >= 3:
                self.boxes.append((self.current_class_id, x1, y1, x2, y2))

    def handle_class_key(self, key: int) -> bool:
        if key not in KEY_TO_CLASS_ID:
            return False
        self.current_class_id = KEY_TO_CLASS_ID[key]
        print(f"Selected class: {self.current_class_id} {class_label(self.current_class_id)}")
        return True

    def run(self):
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win_name, self.on_mouse)

        while self.idx < len(self.images):
            image_path = self.images[self.idx]
            self.image = imread_unicode(image_path)

            if self.image is None:
                print(f"Read failed, skipping: {image_path}")
                self.idx += 1
                continue

            h, w = self.image.shape[:2]
            self.boxes = self.load_existing_boxes(image_path, w, h)

            while True:
                cv2.imshow(self.win_name, self.draw_overlay())
                key = cv2.waitKey(20) & 0xFF

                if self.handle_class_key(key):
                    continue

                if key == ord("s"):
                    self.save_boxes(image_path, w, h)
                    self.idx += 1
                    break

                if key == ord("n"):
                    self.boxes = []
                    self.save_boxes(image_path, w, h)
                    self.idx += 1
                    break

                if key == ord("u") and self.boxes:
                    self.boxes.pop()

                if key == ord("r"):
                    self.boxes = []

                if key == ord("q"):
                    cv2.destroyAllWindows()
                    print("Annotation tool exited.")
                    return

        cv2.destroyAllWindows()
        print("All images annotated.")


def main():
    parser = argparse.ArgumentParser(
        description="Manual YOLO annotation tool for typed foreign objects"
    )
    parser.add_argument("--data_dir", type=Path, default=YOLO_DATA_DIR)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    YoloAnnotator(args.data_dir, args.split, args.start).run()


if __name__ == "__main__":
    main()
