from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task3_alarm.alarm_common import (
    DEFAULT_MODEL_NAME,
    SYSTEM_PROMPT,
    adapt_detection_for_alarm,
    build_alarm_prompt,
    ensure_report_sections,
    read_json,
    write_text,
)
from project_config import ALARM_ADAPTER_DIR, OUTPUTS_DIR


DEFAULT_DETECTION_JSON = OUTPUTS_DIR / "detection.json"
DEFAULT_ADAPTER_DIR = ALARM_ADAPTER_DIR
DEFAULT_OUTPUT_TXT = OUTPUTS_DIR / "alarm_report.txt"


def check_dependencies() -> None:
    missing = []
    for package_name, import_name in [
        ("transformers", "transformers"),
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
        print("任务三推理依赖未安装：")
        print(", ".join(missing))
        print("\n请在 dl_practice 环境中运行：")
        print("pip install transformers datasets peft accelerate safetensors sentencepiece protobuf")
        raise SystemExit(1)


def build_messages(prompt: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def load_lora_model(model_name_or_path: str, adapter_dir: Path, qwen_device: str = "auto"):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"没有找到 LoRA adapter 目录：{adapter_dir}\n"
            "请先运行：python task3_alarm/train_lora_qwen.py"
        )
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"LoRA adapter 缺少 adapter_config.json：{adapter_dir}")
    if not (adapter_dir / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"LoRA adapter 缺少 adapter_model.safetensors：{adapter_dir}")

    try:
        print(f"Qwen 推理设备：{qwen_device}")
        tokenizer_source = adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=True,
            use_fast=False,
        )
        if qwen_device == "auto":
            model_kwargs = {
                "device_map": "auto",
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            }
        else:
            if qwen_device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("当前环境未检测到 CUDA，无法使用 --qwen_device cuda")
            model_kwargs = {
                "torch_dtype": torch.float16 if qwen_device == "cuda" else torch.float32,
            }

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        if qwen_device in {"cpu", "cuda"}:
            model = model.to(qwen_device)
    except Exception as exc:
        print("基础 Qwen 模型或 LoRA adapter 加载失败。")
        print("请确认可以联网下载 Qwen/Qwen2.5-0.5B-Instruct，或使用 --model_name_or_path 指向本地模型路径。")
        print(f"原始错误：{exc}")
        raise SystemExit(1) from exc

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def get_model_device(model):
    return next(model.parameters()).device


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    import torch

    messages = build_messages(prompt)
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    device = get_model_device(model)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return ensure_report_sections(text)


def generate_alarm_report(
    detection_json: Path = DEFAULT_DETECTION_JSON,
    adapter_dir: Path = DEFAULT_ADAPTER_DIR,
    output_txt: Path = DEFAULT_OUTPUT_TXT,
    model_name_or_path: str = DEFAULT_MODEL_NAME,
    top_k: int = 5,
    sort_by: str = "confidence",
    qwen_device: str = "auto",
) -> str:
    check_dependencies()
    detection = read_json(detection_json)
    adapted = adapt_detection_for_alarm(detection, top_k=top_k, sort_by=sort_by)
    prompt = build_alarm_prompt(detection, adapted)
    model, tokenizer = load_lora_model(model_name_or_path, adapter_dir, qwen_device=qwen_device)
    report = generate_text(model, tokenizer, prompt)
    write_text(output_txt, report)
    print(f"报警报告已生成：{output_txt}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate alarm report with Qwen2.5 LoRA adapter")
    parser.add_argument("--detection_json", type=Path, default=DEFAULT_DETECTION_JSON)
    parser.add_argument("--adapter_dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_TXT)
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--sort_by", choices=["confidence", "area"], default="confidence")
    parser.add_argument(
        "--qwen_device",
        choices=["auto", "cuda", "cpu"],
        default="cpu",
        help="Qwen 推理设备，默认 cpu；如需自动使用 GPU 可设为 auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        generate_alarm_report(
            detection_json=args.detection_json,
            adapter_dir=args.adapter_dir,
            output_txt=args.output,
            model_name_or_path=args.model_name_or_path,
            top_k=args.top_k,
            sort_by=args.sort_by,
            qwen_device=args.qwen_device,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
