from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ZIP = PROJECT_ROOT / "data" / "raw" / "yolo_images.zip"
YOLO_DATA_DIR = PROJECT_ROOT / "data" / "yolo_yiwu"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RUN_DIR = PROJECT_ROOT / "runs" / "yolo"

CLASS_NAMES = ["unknown", "stone", "plastic", "metal", "wood"]
CLASS_DISPLAY_NAMES = {
    "stone": "石块异物",
    "plastic": "塑料异物",
    "metal": "金属异物",
    "wood": "木块异物",
    "unknown": "未知异物",
    "yiwu": "未知异物",
}
CLASS_ID_TO_NAME = {i: name for i, name in enumerate(CLASS_NAMES)}
CLASS_NAME_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
