from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RUNS_DIR = PROJECT_ROOT / "runs"


def resolve_project_path(path_value: str | Path | None, default: str | Path | None = None) -> Path:
    raw_value = path_value if path_value not in (None, "") else default
    if raw_value is None:
        raise ValueError("path_value and default cannot both be empty")

    path = Path(os.path.expandvars(str(raw_value))).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def env_path(env_name: str, default: str | Path) -> Path:
    return resolve_project_path(os.getenv(env_name), default)


def env_text(env_name: str, default: str) -> str:
    return os.getenv(env_name, default)


YOLO_DATA_DIR = env_path("YOLO_DATA_DIR", DATA_DIR / "yolo_yiwu")
YOLO_RUN_DIR = env_path("YOLO_RUN_DIR", RUNS_DIR / "yolo")
YOLO_RUN_NAME = env_text("YOLO_RUN_NAME", "yiwu_yolov8s_4class")
YOLO_MODEL_PATH = env_path(
    "YOLO_MODEL_PATH",
    YOLO_RUN_DIR / YOLO_RUN_NAME / "weights" / "best.pt",
)
