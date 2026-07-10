from __future__ import annotations

# 允许直接用 PyCharm/命令行运行本文件，而不必使用 python -m。
import sys
from pathlib import Path as _PathForSys
sys.path.append(str(_PathForSys(__file__).resolve().parents[1]))

import argparse
from datetime import datetime
from pathlib import Path

import torch

from task1_speech.audio_utils import MelFeatureExtractor, load_wav
from task1_speech.speech_config import COMMAND_MEANING, OUTPUT_DIR, RUN_DIR
from task1_speech.speech_model import SpeechCommandTransformer
from utils.json_io import write_json


def load_model(ckpt_path: Path, device: torch.device):
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"找不到模型：{ckpt_path}\n请先运行：python task1_speech/train_speech_transformer.py"
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    model = SpeechCommandTransformer(num_classes=len(ckpt["label_to_index"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_one(wav_path: str | Path, ckpt_path: str | Path = RUN_DIR / "best_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(ckpt_path)
    model, ckpt = load_model(ckpt_path, device)
    feature_extractor = MelFeatureExtractor().to(device)

    waveform, sr = load_wav(wav_path)
    waveform = waveform.unsqueeze(0).to(device)  # [1, 1, samples]
    mel_seq = feature_extractor(waveform)
    logits = model(mel_seq)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    pred_idx = int(torch.argmax(probs).item())

    index_to_label = {int(k): v for k, v in ckpt.get("index_to_label", {}).items()}
    if not index_to_label:
        index_to_label = {v: k for k, v in ckpt["label_to_index"].items()}
    command = str(index_to_label[pred_idx]).lower().strip()
    confidence = float(probs[pred_idx].item())
    meaning = ckpt.get("command_meaning", COMMAND_MEANING).get(command, "未知命令")

    return {
        "command": command,
        "meaning": meaning,
        "confidence": round(confidence, 6),
        "start_detection": command == "go",
        "confirm_alarm": command == "yes",
        "cancel_alarm": command == "no",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(wav_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Predict go/stop/yes/no command from wav and write command.json")
    parser.add_argument("--wav", type=Path, required=True, help="待识别 wav 文件")
    parser.add_argument("--ckpt", type=Path, default=RUN_DIR / "best_model.pt")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "command.json")
    args = parser.parse_args()

    result = predict_one(args.wav, args.ckpt)
    write_json(result, args.output)
    print(result)
    print(f"command.json 已保存：{args.output}")


if __name__ == "__main__":
    main()
