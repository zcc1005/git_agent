from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Dict, Mapping, Optional, Tuple


SkillHandler = Callable[[str, Dict[str, Any]], Dict[str, Any]]
SkillValidator = Callable[[Mapping[str, Any]], Dict[str, Any]]


def _type_matches(value: Any, expected_type: str) -> bool:
    checks = {
        "null": lambda item: item is None,
        "string": lambda item: isinstance(item, str),
        "boolean": lambda item: isinstance(item, bool),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "object": lambda item: isinstance(item, Mapping),
        "array": lambda item: isinstance(item, (list, tuple)),
    }
    checker = checks.get(expected_type)
    return True if checker is None else checker(value)


def _normalize_schema_value(
    value: Any,
    schema: Mapping[str, Any],
    path: str,
    *,
    enforce_required: bool,
) -> Any:
    aliases = schema.get("aliases") or {}
    if isinstance(value, str) and isinstance(aliases, Mapping):
        alias_key = value.strip().lower()
        if alias_key in aliases:
            value = aliases[alias_key]

    expected = schema.get("type")
    expected_types = (
        [str(item) for item in expected]
        if isinstance(expected, (list, tuple))
        else ([str(expected)] if expected else [])
    )
    if expected_types and not any(_type_matches(value, item) for item in expected_types):
        raise ValueError(f"{path} 必须是 {' 或 '.join(expected_types)} 类型")

    allowed_values = schema.get("enum")
    if isinstance(allowed_values, (list, tuple)) and value not in allowed_values:
        choices = "、".join(str(item) for item in allowed_values)
        raise ValueError(f"{path} 只能是：{choices}")

    if isinstance(value, Mapping):
        properties = schema.get("properties") or {}
        if not isinstance(properties, Mapping):
            raise ValueError(f"{path} 的 schema.properties 必须是对象")
        normalized = dict(value)
        if schema.get("additionalProperties") is False:
            unknown = sorted(set(normalized) - set(properties))
            if unknown:
                raise ValueError(f"{path} 包含协议外参数：{', '.join(unknown)}")
        required = set(schema.get("required") or ())
        for name, raw_property_schema in properties.items():
            if not isinstance(raw_property_schema, Mapping):
                continue
            child_path = f"{path}.{name}"
            if name in normalized:
                normalized[name] = _normalize_schema_value(
                    normalized[name],
                    raw_property_schema,
                    child_path,
                    enforce_required=enforce_required,
                )
            elif "default" in raw_property_schema:
                normalized[name] = deepcopy(raw_property_schema["default"])
            elif enforce_required and name in required:
                raise ValueError(f"缺少必填参数：{child_path}")
        return normalized

    if isinstance(value, (list, tuple)):
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if min_items is not None and len(value) < int(min_items):
            raise ValueError(f"{path} 至少需要 {min_items} 项")
        if max_items is not None and len(value) > int(max_items):
            raise ValueError(f"{path} 最多允许 {max_items} 项")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            return [
                _normalize_schema_value(
                    item,
                    item_schema,
                    f"{path}[{index}]",
                    enforce_required=enforce_required,
                )
                for index, item in enumerate(value)
            ]
        return list(value)

    if isinstance(value, str):
        if schema.get("minLength") is not None and len(value.strip()) < int(schema["minLength"]):
            raise ValueError(f"{path} 不能为空")
        if schema.get("maxLength") is not None and len(value) > int(schema["maxLength"]):
            raise ValueError(f"{path} 最长允许 {schema['maxLength']} 个字符")
        value_format = str(schema.get("format") or "")
        try:
            if value_format == "date":
                date.fromisoformat(value)
            elif value_format == "date-time":
                datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            expectation = "YYYY-MM-DD" if value_format == "date" else "ISO 8601 时间"
            raise ValueError(f"{path} 必须是{expectation}") from exc

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        bounds = (
            ("minimum", lambda current, limit: current >= limit, "不小于"),
            ("maximum", lambda current, limit: current <= limit, "不大于"),
            ("exclusiveMinimum", lambda current, limit: current > limit, "大于"),
            ("exclusiveMaximum", lambda current, limit: current < limit, "小于"),
        )
        for keyword, check, description in bounds:
            if keyword in schema and not check(value, schema[keyword]):
                raise ValueError(f"{path} 必须{description} {schema[keyword]}")
    return value


def normalize_schema_arguments(
    arguments: Mapping[str, Any],
    schema: Mapping[str, Any],
    *,
    enforce_required: bool = True,
) -> Dict[str, Any]:
    """Normalize and validate the supported JSON-schema subset used by Skills."""
    normalized = _normalize_schema_value(
        dict(arguments),
        schema,
        "arguments",
        enforce_required=enforce_required,
    )
    if not isinstance(normalized, dict):
        raise ValueError("Skill 顶层参数必须是对象")
    return normalized


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    required_inputs: Tuple[str, ...] = ()
    optional_inputs: Tuple[str, ...] = ()
    safety: str = "read"
    input_schema: Mapping[str, Any] = field(default_factory=dict)

    @property
    def allowed_inputs(self) -> set[str]:
        return set(self.required_inputs) | set(self.optional_inputs)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "description": self.description,
            "required_inputs": list(self.required_inputs),
            "optional_inputs": list(self.optional_inputs),
            "safety": self.safety,
        }
        if self.input_schema:
            result["input_schema"] = deepcopy(dict(self.input_schema))
        return result


@dataclass(frozen=True)
class SkillResult:
    skill_name: str
    ok: bool
    reply: str
    data: Dict[str, Any] = field(default_factory=dict)
    error_code: str = ""
    requires_attachment: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "skill_name": self.skill_name,
            "ok": self.ok,
            "reply": self.reply,
            "data": self.data,
        }
        if self.error_code:
            result["error_code"] = self.error_code
        if self.requires_attachment:
            result["requires_attachment"] = True
        return result


class RuntimeSkill:
    def __init__(
        self,
        spec: SkillSpec,
        handler: SkillHandler,
        validator: Optional[SkillValidator] = None,
    ) -> None:
        self.spec = spec
        self.handler = handler
        self.validator = validator or (lambda values: dict(values))

    def invoke(self, session_id: str, arguments: Mapping[str, Any]) -> SkillResult:
        try:
            supplied = dict(arguments)
            unknown = sorted(set(supplied) - self.spec.allowed_inputs)
            if unknown:
                raise ValueError(f"不支持的参数：{', '.join(unknown)}")
            missing = [
                name
                for name in self.spec.required_inputs
                if supplied.get(name) in (None, "")
            ]
            if missing:
                raise ValueError(f"缺少必填参数：{', '.join(missing)}")
            if self.spec.input_schema:
                supplied = normalize_schema_arguments(
                    supplied,
                    self.spec.input_schema,
                    enforce_required=True,
                )
            normalized = self.validator(supplied)
        except (TypeError, ValueError) as exc:
            needs_attachment = any(
                name in {"image_path", "video_path", "media_path"}
                and dict(arguments).get(name) in (None, "")
                for name in self.spec.required_inputs
            )
            return SkillResult(
                skill_name=self.spec.name,
                ok=False,
                reply=f"参数校验失败：{exc}",
                error_code="invalid_arguments",
                requires_attachment=needs_attachment,
            )

        try:
            raw_result = self.handler(session_id, normalized)
        except (FileNotFoundError, LookupError, ValueError, json.JSONDecodeError) as exc:
            return SkillResult(
                skill_name=self.spec.name,
                ok=False,
                reply=str(exc),
                error_code="execution_failed",
            )
        return SkillResult(
            skill_name=self.spec.name,
            ok=bool(raw_result.get("ok")),
            reply=str(raw_result.get("reply") or ""),
            data=dict(raw_result.get("data") or {}),
            error_code=str(raw_result.get("error_code") or ""),
            requires_attachment=bool(raw_result.get("requires_attachment")),
        )


class SkillRegistry:
    """Closed registry; callers can invoke only explicitly registered skills."""

    def __init__(self) -> None:
        self._skills: Dict[str, RuntimeSkill] = {}

    def register(self, skill: RuntimeSkill) -> None:
        if skill.spec.name in self._skills:
            raise ValueError(f"Skill 已注册：{skill.spec.name}")
        self._skills[skill.spec.name] = skill

    def invoke(
        self,
        skill_name: str,
        *,
        session_id: str,
        arguments: Optional[Mapping[str, Any]] = None,
    ) -> SkillResult:
        skill = self._skills.get(skill_name)
        if skill is None:
            raise LookupError(f"未注册的 Skill：{skill_name}")
        return skill.invoke(session_id, arguments or {})

    def catalog(self) -> list[Dict[str, Any]]:
        return [skill.spec.to_dict() for skill in self._skills.values()]

    def names(self) -> tuple[str, ...]:
        return tuple(self._skills)

    def spec(self, skill_name: str) -> SkillSpec:
        skill = self._skills.get(skill_name)
        if skill is None:
            raise LookupError(f"未注册的 Skill：{skill_name}")
        return skill.spec
