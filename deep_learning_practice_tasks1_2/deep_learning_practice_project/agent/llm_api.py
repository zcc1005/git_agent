from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from project_config import PROJECT_ROOT

from .planners import SkillPlan, SkillPlanner, SkillPlanningError, SkillPlanStep
from .service import AgentService, create_default_service


JSONTransport = Callable[[str, bytes, Mapping[str, str], float], Dict[str, Any]]


class LLMAPIError(SkillPlanningError):
    """A safe, key-free error raised by the remote model connection."""


def load_env_file(path: Path | str) -> bool:
    """Load a small dotenv file without overwriting existing environment values."""
    env_path = Path(path)
    if not env_path.is_file():
        return False
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if value[:1] == value[-1:] and value.startswith(("'", '"')):
            value = value[1:-1]
        if name and name not in os.environ:
            os.environ[name] = value
    return True


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} 必须是 true 或 false")


@dataclass(frozen=True)
class LLMAPIConfig:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: float = 60.0
    max_tokens: int = 1600
    json_mode: bool = True

    @classmethod
    def from_env(cls) -> "LLMAPIConfig":
        api_key = (os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
        model = (os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "").strip()
        base_url = (
            os.getenv("LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ).strip()
        if not api_key:
            raise ValueError("未配置 LLM_API_KEY；请复制 .env.example 为 .env 并填写密钥")
        if not model:
            raise ValueError("未配置 LLM_MODEL；请填写服务商支持的模型名称")
        timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1600"))
        if timeout <= 0:
            raise ValueError("LLM_TIMEOUT_SECONDS 必须大于 0")
        if max_tokens < 128:
            raise ValueError("LLM_MAX_TOKENS 必须至少为 128")
        return cls(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            model=model,
            timeout_seconds=timeout,
            max_tokens=max_tokens,
            json_mode=_env_bool("LLM_JSON_MODE", True),
        )

    @property
    def chat_completions_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"


class OpenAICompatibleClient:
    """Minimal OpenAI-compatible chat client using only Python's standard library."""

    def __init__(
        self,
        config: LLMAPIConfig,
        *,
        transport: Optional[JSONTransport] = None,
    ) -> None:
        self.config = config
        self._transport = transport or self._urlopen_transport

    def complete_json(self, messages: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [dict(message) for message in messages],
            "temperature": 0,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        response = self._transport(
            self.config.chat_completions_url,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers,
            self.config.timeout_seconds,
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMAPIError("大模型响应缺少 choices[0].message.content") from exc
        if isinstance(content, list):
            content = "".join(
                str(item.get("text") or "") if isinstance(item, Mapping) else str(item)
                for item in content
            )
        if not isinstance(content, str) or not content.strip():
            raise LLMAPIError("大模型返回了空内容")
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMAPIError("大模型没有返回合法 JSON") from exc
        if not isinstance(parsed, dict):
            raise LLMAPIError("大模型 JSON 顶层必须是对象")
        return parsed

    @staticmethod
    def _urlopen_transport(
        url: str,
        body: bytes,
        headers: Mapping[str, str],
        timeout: float,
    ) -> Dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=body,
            headers=dict(headers),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            authorization = str(headers.get("Authorization") or "")
            secret = authorization.removeprefix("Bearer ").strip()
            if secret:
                detail = detail.replace(secret, "***")
            raise LLMAPIError(f"大模型 API 返回 HTTP {exc.code}：{detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMAPIError(f"无法连接大模型 API：{exc.reason}") from exc
        except TimeoutError as exc:
            raise LLMAPIError("连接大模型 API 超时") from exc
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMAPIError("大模型 API 返回了非 JSON 响应") from exc
        if not isinstance(result, dict):
            raise LLMAPIError("大模型 API 响应顶层必须是对象")
        return result


class OpenAICompatibleSkillPlanner(SkillPlanner):
    def __init__(self, client: OpenAICompatibleClient) -> None:
        self.client = client

    def plan(
        self,
        message: str,
        *,
        catalog: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
    ) -> SkillPlan:
        safe_context = self._safe_context(context)
        system_prompt = (
            "你是工业皮带异物检测系统的任务规划器。你只负责理解、参数抽取和 Skill 编排，"
            "不得自行判断检测结果、风险等级、报警状态或虚构文件。\n"
            "只可使用提供的封闭 Skill catalog，最多返回 6 个步骤。检测 Skill 已包含确定性风险"
            "研判、历史入库和报警创建，不要重复调用 assess-risk。\n"
            "报警 confirm/cancel 及 review-detection 写操作只有在用户明确要求时才能规划。"
            "control-alarm.action 只能输出 query、confirm、cancel；查看、查询、显示、获取报警状态"
            "一律输出 query，禁止输出 view、show、get、status。"
            "缺少媒体路径、时间范围对应的实际文件或关键参数时，设置 needs_clarification=true。\n"
            "后续步骤可用 $steps.0.data.detection_id 形式引用前一步结果。\n"
            "严格返回 JSON 对象，结构为："
            '{"summary":"简短计划","needs_clarification":false,'
            '"clarification":"","steps":[{"skill_name":"query-history",'
            '"arguments":{}}]}。'
        )
        user_prompt = json.dumps(
            {
                "user_message": message,
                "skill_catalog": list(catalog),
                "context": safe_context,
            },
            ensure_ascii=False,
            default=str,
        )
        payload = self.client.complete_json(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return self._parse_plan(payload, catalog=catalog)

    @staticmethod
    def _safe_context(context: Mapping[str, Any]) -> Dict[str, Any]:
        history = context.get("history") or []
        safe_history = []
        for item in list(history)[-8:]:
            if isinstance(item, Mapping):
                safe_history.append(
                    {
                        "role": str(item.get("role") or ""),
                        "content": str(item.get("content") or "")[:2000],
                    }
                )
        request_context = context.get("request_context")
        if not isinstance(request_context, Mapping):
            request_context = {}
        return {
            "session_id": str(context.get("session_id") or ""),
            "history": safe_history,
            "request_context": dict(request_context),
        }

    @classmethod
    def _parse_plan(
        cls,
        payload: Mapping[str, Any],
        *,
        catalog: Sequence[Mapping[str, Any]] = (),
    ) -> SkillPlan:
        raw_steps = payload.get("steps") or []
        if not isinstance(raw_steps, list):
            raise LLMAPIError("模型计划中的 steps 必须是数组")
        catalog_by_name = {
            str(item.get("name") or ""): item
            for item in catalog
            if isinstance(item, Mapping) and item.get("name")
        }
        steps = []
        for raw_step in raw_steps:
            if not isinstance(raw_step, Mapping):
                raise LLMAPIError("每个 Skill 步骤必须是对象")
            skill_name = str(raw_step.get("skill_name") or "").strip()
            arguments = raw_step.get("arguments") or {}
            if not skill_name:
                raise LLMAPIError("Skill 步骤缺少 skill_name")
            if not isinstance(arguments, Mapping):
                raise LLMAPIError("Skill arguments 必须是对象")
            normalized_arguments = cls._normalize_arguments(
                skill_name,
                arguments,
                catalog_by_name.get(skill_name),
            )
            steps.append(SkillPlanStep(skill_name, normalized_arguments))
        return SkillPlan(
            steps=tuple(steps),
            needs_clarification=bool(payload.get("needs_clarification")),
            clarification=str(payload.get("clarification") or ""),
            summary=str(payload.get("summary") or ""),
        )

    @staticmethod
    def _normalize_arguments(
        skill_name: str,
        arguments: Mapping[str, Any],
        skill_spec: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        normalized = dict(arguments)
        if not isinstance(skill_spec, Mapping):
            return normalized
        input_schema = skill_spec.get("input_schema")
        if not isinstance(input_schema, Mapping):
            return normalized
        properties = input_schema.get("properties") or {}
        if not isinstance(properties, Mapping):
            raise LLMAPIError(f"Skill {skill_name} 的 input_schema.properties 必须是对象")

        if input_schema.get("additionalProperties") is False:
            unknown = sorted(set(normalized) - set(properties))
            if unknown:
                raise LLMAPIError(
                    f"Skill {skill_name} 包含协议外参数：{', '.join(unknown)}"
                )

        for parameter_name, raw_schema in properties.items():
            if not isinstance(raw_schema, Mapping):
                continue
            if parameter_name not in normalized:
                if "default" in raw_schema:
                    normalized[parameter_name] = raw_schema["default"]
                continue

            value = normalized[parameter_name]
            aliases = raw_schema.get("aliases") or {}
            if isinstance(value, str) and isinstance(aliases, Mapping):
                alias_key = value.strip().lower()
                if alias_key in aliases:
                    value = aliases[alias_key]
                    normalized[parameter_name] = value

            allowed_values = raw_schema.get("enum")
            if isinstance(allowed_values, (list, tuple)) and value not in allowed_values:
                choices = ", ".join(str(item) for item in allowed_values)
                raise LLMAPIError(
                    f"Skill {skill_name}.{parameter_name} 只能是：{choices}"
                )

            expected_type = raw_schema.get("type")
            type_checks = {
                "string": lambda item: isinstance(item, str),
                "boolean": lambda item: isinstance(item, bool),
                "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
                "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
                "object": lambda item: isinstance(item, Mapping),
                "array": lambda item: isinstance(item, (list, tuple)),
            }
            checker = type_checks.get(str(expected_type or ""))
            if checker is not None and not checker(value):
                raise LLMAPIError(
                    f"Skill {skill_name}.{parameter_name} 必须是 {expected_type} 类型"
                )
        return normalized


def create_llm_enabled_service(
    db_path: Path | str | None = None,
    *,
    env_file: Path | str | None = None,
    transport: Optional[JSONTransport] = None,
) -> AgentService:
    dotenv_path = Path(env_file) if env_file else PROJECT_ROOT / ".env"
    load_env_file(dotenv_path)
    client = OpenAICompatibleClient(LLMAPIConfig.from_env(), transport=transport)
    planner = OpenAICompatibleSkillPlanner(client)
    planner_mode = os.getenv("LLM_PLANNER_MODE", "hybrid")
    return create_default_service(
        db_path,
        skill_planner=planner,
        skill_planner_mode=planner_mode,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="测试大模型 Skill 规划与执行连接")
    parser.add_argument("message", help="自然语言任务")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--session-id", default="llm-cli")
    parser.add_argument("--db", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--video-start-time")
    parser.add_argument("--line-id")
    args = parser.parse_args()
    service = create_llm_enabled_service(args.db, env_file=args.env_file)
    context: Dict[str, Any] = {}
    if args.image:
        context["image_path"] = str(args.image)
    if args.video:
        context["video_path"] = str(args.video)
    if args.video_start_time:
        context["video_start_time"] = args.video_start_time
    if args.line_id:
        context["line_id"] = args.line_id
    response = service.chat(
        args.message,
        session_id=args.session_id,
        context=context,
    )
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
