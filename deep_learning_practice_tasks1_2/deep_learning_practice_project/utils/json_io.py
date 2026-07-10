import json
from pathlib import Path
from typing import Any, Dict


def write_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 JSON 文件：{path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
