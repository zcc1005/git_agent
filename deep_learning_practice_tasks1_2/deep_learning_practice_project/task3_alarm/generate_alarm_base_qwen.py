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
from task3_alarm.generate_alarm_qwen_lora import check_dependencies


DEFAULT_DETECTION_JSON = PROJECT_ROOT / "outputs" / "detection.json"
DEFAULT_OUTPUT_TXT = PROJECT_ROOT / "outputs" / "alarm_report_base_qwen.txt"


def load_base_model(model_name_or_path: str):
    import torch
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
        print("原始 Qwen2.5-0.5B-Instruct 模型加载失败。")
        print("请确认可以联网下载模型，或使用 --model_name_or_path 指向本地模型路径。")
        print(f"原始错误：{exc}")
        raise SystemExit(1) from exc

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

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


def generate_base_alarm_report(
    detection_json: Path = DEFAULT_DETECTION_JSON,
    output_txt: Path = DEFAULT_OUTPUT_TXT,
    model_name_or_path: str = DEFAULT_MODEL_NAME,
    top_k: int = 5,
    sort_by: str = "confidence",
) -> str:
    check_dependencies()
    detection = read_json(detection_json)
    adapted = adapt_detection_for_alarm(detection, top_k=top_k, sort_by=sort_by)
    prompt = build_alarm_prompt(detection, adapted)
    model, tokenizer = load_base_model(model_name_or_path)
    report = generate_text(model, tokenizer, prompt)
    write_text(output_txt, report)
    print(f"原始 Qwen 报警报告已生成：{output_txt}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate alarm report with base Qwen2.5 model")
    parser.add_argument("--detection_json", type=Path, default=DEFAULT_DETECTION_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_TXT)
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--sort_by", choices=["confidence", "area"], default="confidence")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        generate_base_alarm_report(
            detection_json=args.detection_json,
            output_txt=args.output,
            model_name_or_path=args.model_name_or_path,
            top_k=args.top_k,
            sort_by=args.sort_by,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

