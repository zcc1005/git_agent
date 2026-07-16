from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple


SkillHandler = Callable[[str, Dict[str, Any]], Dict[str, Any]]
SkillValidator = Callable[[Mapping[str, Any]], Dict[str, Any]]


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    required_inputs: Tuple[str, ...] = ()
    optional_inputs: Tuple[str, ...] = ()
    safety: str = "read"

    @property
    def allowed_inputs(self) -> set[str]:
        return set(self.required_inputs) | set(self.optional_inputs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "required_inputs": list(self.required_inputs),
            "optional_inputs": list(self.optional_inputs),
            "safety": self.safety,
        }


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
