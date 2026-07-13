from project_config import OUTPUTS_DIR, PROJECT_ROOT, SPEECH_DATA_DIR, SPEECH_RUN_DIR

DATA_DIR = SPEECH_DATA_DIR
OUTPUT_DIR = OUTPUTS_DIR
RUN_DIR = SPEECH_RUN_DIR

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
