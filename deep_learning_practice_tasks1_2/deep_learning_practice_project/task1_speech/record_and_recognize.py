from __future__ import annotations

# 允许直接用 PyCharm/命令行运行本文件，而不必使用 python -m。
import sys
from pathlib import Path as _PathForSys
sys.path.append(str(_PathForSys(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import sounddevice as sd
import soundfile as sf

from task1_speech.predict_command import predict_one
from task1_speech.speech_config import OUTPUT_DIR, RUN_DIR, SAMPLE_RATE
from utils.json_io import write_json


def record_wav(path: Path, seconds: float = 1.0, sample_rate: int = SAMPLE_RATE):
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"准备录音 {seconds} 秒，请说：go / stop / yes / no")
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    sf.write(str(path), audio, sample_rate)
    print(f"录音已保存：{path}")


def main():
    parser = argparse.ArgumentParser(description="Record microphone audio and recognize command")
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--ckpt", type=Path, default=RUN_DIR / "best_model.pt")
    parser.add_argument("--wav_out", type=Path, default=OUTPUT_DIR / "recorded_command.wav")
    parser.add_argument("--json_out", type=Path, default=OUTPUT_DIR / "command.json")
    args = parser.parse_args()

    record_wav(args.wav_out, args.seconds)
    result = predict_one(args.wav_out, args.ckpt)
    write_json(result, args.json_out)
    print(result)
    print(f"command.json 已保存：{args.json_out}")


if __name__ == "__main__":
    main()
