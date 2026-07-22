from __future__ import annotations

from typing import Any, Callable, Dict

from .intents import Intent, IntentMatch


ToolHandler = Callable[[str, Dict[str, Any]], Dict[str, Any]]


class ToolRouter:
    """Maps recognized intents to named, independently testable tools."""

    def __init__(self) -> None:
        self._routes: Dict[Intent, tuple[str, ToolHandler]] = {}

    def register(self, intent: Intent, tool_name: str, handler: ToolHandler) -> None:
        if intent in {Intent.UNKNOWN, Intent.HELP}:
            raise ValueError(f"{intent.value} 由会话层处理，不能注册为业务工具")
        self._routes[intent] = (tool_name, handler)

    def dispatch(
        self,
        match: IntentMatch,
        session_id: str,
        context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        route = self._routes.get(match.intent)
        if route is None:
            raise LookupError(f"意图未注册工具：{match.intent.value}")
        tool_name, handler = route
        tool_context = dict(context or {})
        for key, value in match.slots.items():
            tool_context.setdefault(key, value)
        result = handler(session_id, tool_context)
        return {"tool_name": tool_name, **result}
