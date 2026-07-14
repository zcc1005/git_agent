from project_config import OUTPUTS_DIR, PROJECT_ROOT, YOLO_DATA_DIR, YOLO_RUN_DIR

RAW_ZIP = PROJECT_ROOT / "data" / "raw" / "yolo_images.zip"
OUTPUT_DIR = OUTPUTS_DIR
RUN_DIR = YOLO_RUN_DIR

CLASS_NAMES = ["stone", "plastic", "metal", "wood"]
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
