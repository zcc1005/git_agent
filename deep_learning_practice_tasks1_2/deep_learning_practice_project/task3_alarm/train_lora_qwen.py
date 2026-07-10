from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task3_alarm.alarm_common import DEFAULT_MODEL_NAME, SYSTEM_PROMPT


DATASET_PATH = PROJECT_ROOT / "task3_alarm" / "alarm_train_100.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "task3_alarm" / "qwen_alarm_lora"


def check_dependencies() -> None:
    missing = []
    for package_name, import_name in [
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("peft", "peft"),
        ("accelerate", "accelerate"),
        ("safetensors", "safetensors"),
        ("sentencepiece", "sentencepiece"),
        ("protobuf", "google.protobuf"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        print("任务三 LoRA 微调依赖未安装：")
        print(", ".join(missing))
        print("\n请在 dl_practice 环境中运行：")
        print("pip install transformers datasets peft accelerate safetensors sentencepiece protobuf")
        raise SystemExit(1)


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"没有找到训练集：{path}")

    samples: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_no} 行 JSON 解析失败：{exc.msg}") from exc

            for field in ("instruction", "input", "output"):
                if field not in sample:
                    raise ValueError(f"第 {line_no} 行缺少字段：{field}")
                if not isinstance(sample[field], str):
                    raise ValueError(f"第 {line_no} 行字段 {field} 必须是字符串")
            samples.append(sample)

    if not samples:
        raise ValueError(f"训练集为空：{path}")
    return samples


def format_messages(sample: Dict[str, str]) -> List[Dict[str, str]]:
    user_content = sample["instruction"].strip() + "\n\n检测JSON如下：\n" + sample["input"].strip()
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": sample["output"].strip()},
    ]


class AlarmSftDataset:
    def __init__(self, samples: List[Dict[str, str]], tokenizer: Any, max_seq_length: int) -> None:
        import torch

        self.items = []
        self.torch = torch
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        for idx, sample in enumerate(samples, start=1):
            encoded = self.encode_sample(sample)
            labels = encoded["labels"]
            if all(value == -100 for value in labels):
                print(f"警告：第 {idx} 条样本被截断后没有 assistant 标签，已跳过。")
                continue
            self.items.append(encoded)

        if not self.items:
            raise ValueError(
                "所有样本都没有有效训练标签。请增大 max_seq_length，或检查训练数据格式。"
            )

    def encode_sample(self, sample: Dict[str, str]) -> Dict[str, List[int]]:
        messages = format_messages(sample)
        prompt_messages = messages[:2]

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        full = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_attention_mask=True,
            add_special_tokens=False,
        )
        prompt = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_seq_length,
            add_special_tokens=False,
        )

        input_ids = list(full["input_ids"])
        attention_mask = list(full["attention_mask"])
        labels = input_ids.copy()
        prompt_len = min(len(prompt["input_ids"]), self.max_seq_length)

        for i in range(prompt_len):
            labels[i] = -100
        pad_token_id = self.tokenizer.pad_token_id
        for i, mask in enumerate(attention_mask):
            if mask == 0 or input_ids[i] == pad_token_id:
                labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.items[index]
        return {
            key: self.torch.tensor(value, dtype=self.torch.long)
            for key, value in item.items()
        }


def load_model_and_tokenizer(model_name_or_path: str, gradient_checkpointing: bool):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            use_fast=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    except Exception as exc:
        print("Qwen2.5-0.5B-Instruct 模型加载失败。")
        print("请确认电脑可以联网下载模型，或已将模型提前下载到本地路径并用 --model_name_or_path 指定。")
        print(f"原始错误：{exc}")
        raise SystemExit(1) from exc

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def save_training_visualizations(
    log_history: List[Dict[str, Any]],
    output_dir: Path,
    run_config: Dict[str, Any],
    final_metrics: Dict[str, Any],
) -> None:
    metrics_dir = output_dir / "training_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    run_tag = (
        f"bs{run_config['per_device_train_batch_size']}_"
        f"ga{run_config['gradient_accumulation_steps']}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    history_path = metrics_dir / f"training_log_history_{run_tag}.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_config": run_config,
                "final_metrics": final_metrics,
                "log_history": log_history,
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    if log_history:
        csv_path = metrics_dir / f"training_log_history_{run_tag}.csv"
        fieldnames = sorted({key for item in log_history for key in item.keys()})
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(log_history)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped saving training metrics plot")
        print(f"Training logs saved to: {history_path}")
        return

    plot_specs = [
        ("loss", "Training Loss"),
        ("eval_loss", "Evaluation Loss"),
        ("learning_rate", "Learning Rate"),
        ("grad_norm", "Gradient Norm"),
    ]
    available_specs = []
    for metric_key, title in plot_specs:
        series = []
        for index, item in enumerate(log_history):
            value = item.get(metric_key)
            if isinstance(value, (int, float)):
                x_value = item.get("step", item.get("epoch", index + 1))
                series.append((x_value, value))
        if series:
            available_specs.append((metric_key, title, series))

    if not available_specs:
        print(f"Training logs saved to: {history_path}")
        print("No numeric training metrics found for plotting.")
        return

    fig, axes = plt.subplots(
        len(available_specs),
        1,
        figsize=(9, max(3, 2.8 * len(available_specs))),
        squeeze=False,
    )
    fig.suptitle(
        "Training Metrics "
        f"(batch_size={run_config['per_device_train_batch_size']}, "
        f"grad_accum={run_config['gradient_accumulation_steps']})"
    )

    for ax, (_, title, series) in zip(axes.flat, available_specs):
        xs = [point[0] for point in series]
        ys = [point[1] for point in series]
        ax.plot(xs, ys, marker="o", linewidth=1.6, markersize=3)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    image_path = metrics_dir / f"training_metrics_{run_tag}.png"
    fig.savefig(image_path, dpi=160)
    plt.close(fig)

    print(f"Training logs saved to: {history_path}")
    print(f"Training metrics plot saved to: {image_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Qwen alarm report generation")
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_NAME, help="基础模型名称或本地路径")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="报警报告 jsonl 训练集")
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR, help="LoRA adapter 输出目录")
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    return parser.parse_args()


def main() -> None:
    check_dependencies()

    import torch
    from transformers import Trainer, TrainingArguments

    args = parse_args()
    samples = load_jsonl(args.dataset)
    print(f"读取训练样本：{len(samples)} 条")
    print(f"基础模型：{args.model_name_or_path}")

    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )
    train_dataset = AlarmSftDataset(samples, tokenizer, max_seq_length=args.max_seq_length)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=torch.cuda.is_available(),
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        optim="adamw_torch",
    )

    import inspect

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
    }

    trainer_signature = inspect.signature(Trainer.__init__)

    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    try:
        train_result = trainer.train()
    except torch.cuda.OutOfMemoryError as exc:
        print("GPU 显存不足，LoRA 微调中断。")
        print("建议将 --max_seq_length 调低到 768 或 512；batch size 保持 1，必要时降低 gradient_accumulation_steps。")
        raise SystemExit(1) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_training_visualizations(
        log_history=trainer.state.log_history,
        output_dir=args.output_dir,
        run_config={
            "model_name_or_path": args.model_name_or_path,
            "dataset": str(args.dataset),
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "max_seq_length": args.max_seq_length,
        },
        final_metrics=train_result.metrics,
    )
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("LoRA 微调完成")
    print(f"LoRA adapter 已保存：{args.output_dir}")


if __name__ == "__main__":
    main()
