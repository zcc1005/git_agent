from __future__ import annotations

# 允许直接用 PyCharm/命令行运行本文件，而不必使用 python -m。
import sys
import os
from pathlib import Path as _PathForSys
sys.path.append(str(_PathForSys(__file__).resolve().parents[1]))

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from task1_speech.audio_utils import MelFeatureExtractor, fix_audio_length
from task1_speech.speech_config import (
    COMMAND_MEANING,
    DATA_DIR,
    INDEX_TO_LABEL,
    LABEL_TO_INDEX,
    RUN_DIR,
    SAMPLE_RATE,
)
from task1_speech.speech_model import SpeechCommandTransformer


class FilteredSpeechCommands(Dataset):
    """只保留 go/stop/yes/no 四类语音命令。"""

    def __init__(self, root: Path, subset: str):
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=str(root),
            download=False,
            subset=subset,
        )
        self.indices: List[int] = []
        for i, file_path in enumerate(self.dataset._walker):
            label = Path(file_path).parent.name
            if label in LABEL_TO_INDEX:
                self.indices.append(i)
        if len(self.indices) == 0:
            raise RuntimeError("没有找到 go/stop/yes/no 样本，请检查 Google Speech Commands 下载是否完整。")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        waveform, sample_rate, label, *_ = self.dataset[self.indices[idx]]
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
        waveform = fix_audio_length(waveform)
        target = LABEL_TO_INDEX[label]
        return waveform, target


def build_dataloaders(root: Path, batch_size: int, num_workers: int):
    os.makedirs(root, exist_ok=True)
    train_set = FilteredSpeechCommands(root, subset="training")
    val_set = FilteredSpeechCommands(root, subset="validation")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def run_one_epoch(model, feature_extractor, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, total_correct, total_count = 0.0, 0, 0
    desc = "train" if train else "valid"

    with torch.set_grad_enabled(train):
        for waveforms, targets in tqdm(loader, desc=desc, leave=True):
            waveforms = waveforms.to(device)
            targets = targets.to(device)
            mels = feature_extractor(waveforms)
            logits = model(mels)
            loss = criterion(logits, targets)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            preds = logits.argmax(dim=1)
            total_loss += float(loss.item()) * targets.size(0)
            total_correct += int((preds == targets).sum().item())
            total_count += int(targets.size(0))

    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


@torch.no_grad()
def evaluate_predictions(model, feature_extractor, loader, device) -> Tuple[List[int], List[int]]:
    model.eval()
    all_targets: List[int] = []
    all_preds: List[int] = []

    for waveforms, targets in tqdm(loader, desc="eval", leave=True):
        waveforms = waveforms.to(device)
        targets = targets.to(device)
        mels = feature_extractor(waveforms)
        logits = model(mels)
        preds = logits.argmax(dim=1)
        all_targets.extend(targets.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    return all_targets, all_preds


def build_confusion_matrix(targets: List[int], preds: List[int], num_classes: int) -> List[List[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for target, pred in zip(targets, preds):
        matrix[int(target)][int(pred)] += 1
    return matrix


def build_classification_report(matrix: List[List[int]], labels: List[str]) -> Dict[str, object]:
    per_class = {}
    total_correct = 0
    total_count = 0

    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(matrix[row][i] for row in range(len(labels)) if row != i)
        fn = sum(matrix[i][col] for col in range(len(labels)) if col != i)
        support = sum(matrix[i])
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }
        total_correct += tp
        total_count += support

    return {
        "accuracy": round(total_correct / max(total_count, 1), 6),
        "labels": labels,
        "per_class": per_class,
    }


def _import_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:
        raise RuntimeError("缺少绘图依赖，请先安装：pip install matplotlib") from exc


def save_training_curves(history: List[Dict[str, float]], out_dir: Path) -> Tuple[Path, Path]:
    plt = _import_pyplot()
    epochs = [row["epoch"] for row in history]

    loss_path = out_dir / "loss_curve.png"
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train loss")
    plt.plot(epochs, [row["val_loss"] for row in history], marker="s", label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Speech Transformer Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()

    acc_path = out_dir / "accuracy_curve.png"
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(epochs, [row["train_acc"] for row in history], marker="o", label="Train acc")
    plt.plot(epochs, [row["val_acc"] for row in history], marker="s", label="Val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Speech Transformer Accuracy")
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_path)
    plt.close()

    return loss_path, acc_path


def save_confusion_matrix_plot(matrix: List[List[int]], labels: List[str], out_dir: Path) -> Path:
    plt = _import_pyplot()
    cm_path = out_dir / "confusion_matrix.png"

    plt.figure(figsize=(6, 5), dpi=150)
    image = plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("Speech Transformer Confusion Matrix")
    plt.colorbar(image, fraction=0.046, pad=0.04)
    tick_positions = list(range(len(labels)))
    plt.xticks(tick_positions, labels)
    plt.yticks(tick_positions, labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    max_value = max([max(row) for row in matrix], default=0)
    threshold = max_value / 2.0
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            color = "white" if value > threshold else "black"
            plt.text(j, i, str(value), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    return cm_path


def save_experiment_results(
    model,
    feature_extractor,
    val_loader,
    device,
    out_dir: Path,
    history: List[Dict[str, float]],
) -> None:
    labels = [INDEX_TO_LABEL[i] for i in range(len(INDEX_TO_LABEL))]
    loss_path, acc_path = save_training_curves(history, out_dir)

    targets, preds = evaluate_predictions(model, feature_extractor, val_loader, device)
    matrix = build_confusion_matrix(targets, preds, len(labels))
    report = build_classification_report(matrix, labels)
    cm_path = save_confusion_matrix_plot(matrix, labels, out_dir)

    matrix_json = out_dir / "confusion_matrix.json"
    with matrix_json.open("w", encoding="utf-8") as f:
        json.dump({"labels": labels, "matrix": matrix}, f, ensure_ascii=False, indent=2)

    report_json = out_dir / "classification_report.json"
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"训练 loss 曲线已保存：{loss_path}")
    print(f"训练准确率曲线已保存：{acc_path}")
    print(f"混淆矩阵图片已保存：{cm_path}")
    print(f"混淆矩阵数据已保存：{matrix_json}")
    print(f"分类指标已保存：{report_json}")


def main():
    parser = argparse.ArgumentParser(description="Train Transformer on Google Speech Commands go/stop/yes/no")
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out_dir", type=Path, default=RUN_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0, help="Windows/PyCharm 下建议保持 0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders(args.data_dir, args.batch_size, args.num_workers)
    feature_extractor = MelFeatureExtractor().to(device)
    model = SpeechCommandTransformer(num_classes=len(LABEL_TO_INDEX)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_one_epoch(model, feature_extractor, train_loader, criterion, optimizer, device, True)
        val_loss, val_acc = run_one_epoch(model, feature_extractor, val_loader, criterion, optimizer, device, False)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)
        tqdm.write(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc >= best_acc:
            best_acc = val_acc
            best_epoch = epoch
            ckpt = {
                "model_state": model.state_dict(),
                "label_to_index": LABEL_TO_INDEX,
                "index_to_label": INDEX_TO_LABEL,
                "command_meaning": COMMAND_MEANING,
                "sample_rate": SAMPLE_RATE,
                "n_mels": 64,
                "best_val_acc": best_acc,
                "best_epoch": best_epoch,
            }
            torch.save(ckpt, args.out_dir / "best_model.pt")

    with (args.out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    with (args.out_dir / "command_schema.json").open("w", encoding="utf-8") as f:
        json.dump(
            {k: {"index": v, "meaning": COMMAND_MEANING[k]} for k, v in LABEL_TO_INDEX.items()},
            f,
            ensure_ascii=False,
            indent=2,
        )

    best_ckpt_path = args.out_dir / "best_model.pt"
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    save_experiment_results(
        model=model,
        feature_extractor=feature_extractor,
        val_loader=val_loader,
        device=device,
        out_dir=args.out_dir,
        history=history,
    )

    print(f"训练完成，最佳验证准确率：{best_acc:.4f}")
    print(f"最佳 epoch：{best_epoch}")
    print(f"模型已保存：{best_ckpt_path}")


if __name__ == "__main__":
    main()
