from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "task3_alarm" / "alarm_train_100.jsonl"
REQUIRED_FIELDS = {"instruction", "input", "output"}


def load_and_check_dataset(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"没有找到训练集文件：{path}")

    samples: List[Dict[str, Any]] = []
    errors: List[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"第 {line_no} 行 JSON 解析失败：{exc.msg}")
                continue

            missing = REQUIRED_FIELDS - set(sample.keys())
            if missing:
                errors.append(f"第 {line_no} 行缺少字段：{sorted(missing)}")
                continue

            for field in REQUIRED_FIELDS:
                if not isinstance(sample.get(field), str):
                    errors.append(f"第 {line_no} 行字段 {field} 必须是字符串")
                    break
            else:
                samples.append(sample)

    if errors:
        print("数据集检查发现问题：")
        for error in errors[:20]:
            print(f"- {error}")
        if len(errors) > 20:
            print(f"... 还有 {len(errors) - 20} 个问题未显示")
        raise SystemExit(1)

    return samples


def collect_class_distribution(samples: List[Dict[str, Any]]) -> Counter:
    counter: Counter = Counter()
    for idx, sample in enumerate(samples, start=1):
        try:
            detection_input = json.loads(sample["input"])
        except json.JSONDecodeError as exc:
            raise ValueError(f"第 {idx} 条样本 input 字段不是合法 JSON 字符串：{exc.msg}") from exc

        objects = detection_input.get("objects", [])
        if not isinstance(objects, list):
            raise ValueError(f"第 {idx} 条样本 input.objects 必须是列表")

        for obj in objects:
            if not isinstance(obj, dict):
                continue
            class_name = obj.get("class_name") or obj.get("class") or "unknown"
            counter[str(class_name)] += 1
    return counter


def collect_risk_distribution(samples: List[Dict[str, Any]]) -> Counter:
    counter: Counter = Counter()
    pattern = re.compile(r"(高风险|中风险|低风险|无风险)")
    for sample in samples:
        matches = pattern.findall(sample["output"])
        if matches:
            for item in matches:
                counter[item] += 1
        else:
            counter["未标注风险等级"] += 1
    return counter


def main() -> None:
    samples = load_and_check_dataset(DATASET_PATH)
    class_counter = collect_class_distribution(samples)
    risk_counter = collect_risk_distribution(samples)

    print(f"训练集路径：{DATASET_PATH}")
    print(f"样本总数：{len(samples)}")

    print("\n异物类别分布：")
    for name, count in class_counter.most_common():
        print(f"- {name}: {count}")

    print("\n风险等级分布：")
    for name, count in risk_counter.most_common():
        print(f"- {name}: {count}")

    print("\n前 1 条样本：")
    if samples:
        print(json.dumps(samples[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

