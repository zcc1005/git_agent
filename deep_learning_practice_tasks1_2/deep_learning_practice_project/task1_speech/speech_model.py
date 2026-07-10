import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """正弦位置编码，用于给语音帧序列加入时间顺序信息。"""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, d_model]
        return x + self.pe[:, : x.size(1), :]


class SpeechCommandTransformer(nn.Module):
    """
    wav -> Mel Spectrogram -> Linear Embedding -> Transformer Encoder -> Pooling -> FC.
    本任务是 Sequence -> Class，所以只使用 Encoder，不使用 Decoder。
    """
    def __init__(
        self,
        n_mels: int = 64,
        num_classes: int = 4,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_mels, d_model)
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=512)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, mel_seq: torch.Tensor) -> torch.Tensor:
        # mel_seq: [batch, time, n_mels]
        x = self.input_proj(mel_seq)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits
