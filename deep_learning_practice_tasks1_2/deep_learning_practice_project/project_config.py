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
YOLO_RUN_NAME = env_text("YOLO_RUN_NAME", "yiwu_yolov8n")
YOLO_MODEL_PATH = env_path(
    "YOLO_MODEL_PATH",
    YOLO_RUN_DIR / YOLO_RUN_NAME / "weights" / "best.pt",
)

SPEECH_DATA_DIR = env_path("SPEECH_DATA_DIR", DATA_DIR / "speech_commands")
SPEECH_RUN_DIR = env_path("SPEECH_RUN_DIR", RUNS_DIR / "speech_transformer")
SPEECH_CKPT_PATH = env_path("SPEECH_CKPT_PATH", SPEECH_RUN_DIR / "best_model.pt")

ALARM_DATASET_PATH = env_path("ALARM_DATASET_PATH", PROJECT_ROOT / "task3_alarm" / "alarm_train_100.jsonl")
ALARM_ADAPTER_DIR = env_path("ALARM_ADAPTER_DIR", OUTPUTS_DIR / "task3_alarm" / "qwen_alarm_lora")
QWEN_MODEL_NAME = env_text("QWEN_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
