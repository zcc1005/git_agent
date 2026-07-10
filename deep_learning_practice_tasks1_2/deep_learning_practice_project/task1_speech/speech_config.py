from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "speech_commands"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RUN_DIR = PROJECT_ROOT / "runs" / "speech_transformer"

LABEL_TO_INDEX = {
    "go": 0,
    "stop": 1,
    "yes": 2,
    "no": 3,
}

INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}

COMMAND_MEANING = {
    "go": "开始检测",
    "stop": "停止检测",
    "yes": "确认报警",
    "no": "取消报警",
}

SAMPLE_RATE = 16000
NUM_SAMPLES = 16000
