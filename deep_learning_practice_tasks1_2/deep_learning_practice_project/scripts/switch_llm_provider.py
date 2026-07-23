from __future__ import annotations

import argparse
from pathlib import Path


PROVIDERS = {
    "deepseek": {
        "key_names": ("LLM_DEEPSEEK_API_KEY", "OPENAI_API_KEY"),
        "model_names": ("LLM_DEEPSEEK_MODEL",),
        "default_model": "deepseek-v4-pro",
    },
    "c4ai": {
        "key_names": ("LLM_C4AI_API_KEY", "LLM_API_KEY"),
        "model_names": ("LLM_C4AI_MODEL",),
        "default_model": "jiaorong-deepseek-v4-pro",
    },
}


def _dotenv_values(content: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        values[name.strip()] = value.strip().strip("\"'")
    return values


def _first_value(values: dict[str, str], names: tuple[str, ...]) -> str:
    return next(
        (values.get(name, "").strip() for name in names if values.get(name, "").strip()),
        "",
    )


def switch_provider(env_path: Path, provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized not in PROVIDERS:
        raise ValueError("供应商只能是 deepseek 或 c4ai")
    if not env_path.is_file():
        raise FileNotFoundError(f"没有找到 {env_path}，请先从 .env.example 创建 .env")

    content = env_path.read_text(encoding="utf-8")
    values = _dotenv_values(content)
    profile = PROVIDERS[normalized]
    if not _first_value(values, profile["key_names"]):
        names = " 或 ".join(profile["key_names"])
        raise ValueError(f"{normalized} 配置缺少 {names}，请先在 .env 中填写密钥")

    lines = content.splitlines()
    replaced = False
    for index, line in enumerate(lines):
        if line.strip().startswith("LLM_PROVIDER="):
            lines[index] = f"LLM_PROVIDER={normalized}"
            replaced = True
            break
    if not replaced:
        lines.insert(0, f"LLM_PROVIDER={normalized}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return (
        _first_value(values, profile["model_names"])
        or str(profile["default_model"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="切换智能体使用的大模型供应商")
    parser.add_argument("provider", choices=tuple(PROVIDERS))
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".env",
    )
    args = parser.parse_args()
    try:
        model = switch_provider(args.env_file, args.provider)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(f"已切换到 {args.provider}（模型：{model}）。")
    print("请重启 Web 服务，使新配置生效。")


if __name__ == "__main__":
    main()
