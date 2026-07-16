from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Protocol, Sequence, runtime_checkable


class SkillPlanningError(RuntimeError):
    """Raised when a model response cannot be used as a safe Skill plan."""


@dataclass(frozen=True)
class SkillPlanStep:
    skill_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"skill_name": self.skill_name, "arguments": self.arguments}


@dataclass(frozen=True)
class SkillPlan:
    steps: tuple[SkillPlanStep, ...] = ()
    needs_clarification: bool = False
    clarification: str = ""
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "needs_clarification": self.needs_clarification,
            "clarification": self.clarification,
            "summary": self.summary,
        }


@runtime_checkable
class SkillPlanner(Protocol):
    def plan(
        self,
        message: str,
        *,
        catalog: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
    ) -> SkillPlan:
        """Convert natural language into a closed, structured Skill plan."""
