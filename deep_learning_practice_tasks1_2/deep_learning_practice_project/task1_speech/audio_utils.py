from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torchaudio

from task1_speech.speech_config import NUM_SAMPLES, SAMPLE_RATE


def fix_audio_length(waveform: torch.Tensor, num_samples: int = NUM_SAMPLES) -> torch.Tensor:
    """把音频统一裁剪/补零到 1 秒，便于 Transformer 处理固定长度输入。"""
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    current = waveform.size(1)
    if current > num_samples:
        waveform = waveform[:, :num_samples]
    elif current < num_samples:
        pad = num_samples - current
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    return waveform


def load_wav(path: str | Path) -> Tuple[torch.Tensor, int]:
    waveform, sr = torchaudio.load(str(path))
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        sr = SAMPLE_RATE
    waveform = fix_audio_length(waveform)
    return waveform, sr


class MelFeatureExtractor(torch.nn.Module):
    """将原始波形转换为 Transformer 需要的 [time, n_mels] 序列。"""

    def __init__(self, sample_rate: int = SAMPLE_RATE, n_mels: int = 64):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=n_mels,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [batch, 1, samples] or [1, samples]
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        b, c, t = waveform.shape
        waveform = waveform.reshape(b * c, t)
        mel = self.mel(waveform)       # [batch, n_mels, time]
        mel = self.to_db(mel)
        mel = (mel - mel.mean(dim=(1, 2), keepdim=True)) / (mel.std(dim=(1, 2), keepdim=True) + 1e-6)
        mel = mel.transpose(1, 2)      # [batch, time, n_mels]
        return mel
