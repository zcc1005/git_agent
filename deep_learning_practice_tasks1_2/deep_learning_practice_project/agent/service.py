from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from project_config import OUTPUTS_DIR
from storage import SQLiteHistoryStore

from .intents import Intent
from .intents import RuleBasedIntentRecognizer
from .planners import SkillPlan, SkillPlanner, SkillPlanningError
from .recognizers import (
    HybridIntentRecognizer,
    IntentRecognizer,
    RecognitionMode,
)
from .router import ToolRouter
from .skills import SkillRegistry, create_builtin_skill_registry
from .tools import AgentTools


HELP_TEXT = (
    "我支持：检测这张图片、检测这段视频、查询上一轮结果、统计今天高风险报警次数、"
    "生成今日风险报告、确认报警、取消报警。还可通过 Skill 接口执行风险研判、"
    "按时间/风险/线路查询、人工复核和组合巡检任务。"
)
MISSING_MEDIA_REPLY = "还没有看到图片/视频，发过来立刻帮你分析。"
ATTACHMENT_LABELS = {"image": "图片", "video": "视频"}

PLANNING_HINT = re.compile(
    r"(?:\d{1,2}[：:]\d{2}.*(?:-|到|至).*\d{1,2}[：:]\d{2}|"
    r"今天.*(?:上午|下午|早上|晚上).*(?:-|到|至)|ROI|抽帧|阈值|线路|"
    r"风险研判|人工复核|误报|假阳性|闭环|处理完成|"
    r"并且|然后|同时|随后|再生成|以及)",
    re.I,
)


class AgentService:
    """Application facade used by CLI tests and the future Flask endpoint."""

    def __init__(
        self,
        store: Optional[SQLiteHistoryStore] = None,
        *,
        tools: Optional[AgentTools] = None,
        recognizer: Optional[IntentRecognizer] = None,
        model_recognizer: Optional[IntentRecognizer] = None,
        recognition_mode: RecognitionMode | str = RecognitionMode.HYBRID,
        model_confidence_threshold: float = 0.75,
        skill_registry: Optional[SkillRegistry] = None,
        skill_planner: Optional[SkillPlanner] = None,
        skill_planner_mode: str = "hybrid",
    ) -> None:
        if recognizer is not None and model_recognizer is not None:
            raise ValueError("recognizer 与 model_recognizer 不能同时提供")
        self.store = store or (
            tools.store
            if tools is not None
            else SQLiteHistoryStore(OUTPUTS_DIR / "agent_history.sqlite3")
        )
        self.tools = tools or AgentTools(self.store)
        self.skill_registry = skill_registry or create_builtin_skill_registry(self.tools)
        self.skill_planner = skill_planner
        self.skill_planner_mode = skill_planner_mode.strip().lower()
        if self.skill_planner_mode not in {"hybrid", "always"}:
            raise ValueError("skill_planner_mode 只能是 hybrid 或 always")
        self.recognizer = recognizer or HybridIntentRecognizer(
            model_recognizer=model_recognizer,
            mode=recognition_mode,
            model_confidence_threshold=model_confidence_threshold,
        )
        self.router = ToolRouter()
        self.router.register(
            Intent.DETECT_IMAGE,
            "image_detection",
            self._skill_handler("detect-image"),
        )
        self.router.register(
            Intent.DETECT_VIDEO,
            "video_detection",
            self._skill_handler("detect-video"),
        )
        self.router.register(Intent.PREVIOUS_RESULT, "history_query", self.tools.previous_result)
        self.router.register(
            Intent.COUNT_HIGH_RISK_TODAY,
            "high_risk_counter",
            self.tools.count_high_risk_today,
        )
        self.router.register(
            Intent.GENERATE_DAILY_REPORT,
            "daily_risk_report",
            self.tools.generate_daily_report,
        )
        self.router.register(
            Intent.CURRENT_ALARM,
            "alarm_control",
            self._skill_handler("control-alarm", action="query"),
        )
        self.router.register(
            Intent.CONFIRM_ALARM,
            "alarm_control",
            self._skill_handler("control-alarm", action="confirm"),
        )
        self.router.register(
            Intent.CANCEL_ALARM,
            "alarm_control",
            self._skill_handler("control-alarm", action="cancel"),
        )

    def _skill_handler(self, skill_name: str, **defaults: Any):
        def handler(session_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
            arguments = {**defaults, **context}
            return self.skill_registry.invoke(
                skill_name,
                session_id=session_id,
                arguments=arguments,
            ).to_dict()

        return handler

    def run_skill(
        self,
        skill_name: str,
        *,
        session_id: str = "default",
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Stable deterministic entry point for a future LLM planner or Web API."""
        return self.skill_registry.invoke(
            skill_name,
            session_id=session_id,
            arguments=arguments or {},
        ).to_dict()

    def skill_catalog(self) -> list[Dict[str, Any]]:
        return self.skill_registry.catalog()

    @staticmethod
    def _attachment_metadata(context: Mapping[str, Any]) -> Dict[str, Any]:
        for media_type, path_key in (("image", "image_path"), ("video", "video_path")):
            raw_path = context.get(path_key)
            if not raw_path:
                continue
            path = str(raw_path)
            attachment = {
                "media_type": media_type,
                "path": path,
                "file_name": Path(path).name,
            }
            for name in ("video_start_time", "line_id"):
                if context.get(name) not in (None, ""):
                    attachment[name] = context[name]
            preview = context.get("_attachment_preview")
            if isinstance(preview, Mapping):
                for name in ("preview_path", "poster_path"):
                    if preview.get(name) not in (None, ""):
                        attachment[name] = str(preview[name])
            return attachment
        return {}

    def _latest_attachment_context(self, session_id: str) -> Dict[str, Any]:
        for item in reversed(self.store.list_messages(session_id, limit=50)):
            metadata = item.get("metadata")
            if not isinstance(metadata, Mapping):
                continue
            attachment = metadata.get("attachment")
            if not isinstance(attachment, Mapping):
                continue
            media_type = str(attachment.get("media_type") or "")
            path = str(attachment.get("path") or "")
            if media_type not in ATTACHMENT_LABELS or not path or not Path(path).is_file():
                continue
            context: Dict[str, Any] = {f"{media_type}_path": path}
            for name in ("video_start_time", "line_id"):
                if attachment.get(name) not in (None, ""):
                    context[name] = attachment[name]
            return context
        return {}

    def receive_attachment(
        self,
        media_type: str,
        media_path: str | Path,
        *,
        session_id: str = "default",
        original_name: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist an uploaded attachment so a later chat turn can use it."""
        normalized_type = media_type.strip().lower()
        if normalized_type not in ATTACHMENT_LABELS:
            raise ValueError("media_type 只能是 image 或 video")
        path_key = f"{normalized_type}_path"
        attachment_context = dict(context or {})
        attachment_context[path_key] = str(media_path)
        attachment = self._attachment_metadata(attachment_context)
        if original_name.strip():
            attachment["file_name"] = original_name.strip()

        label = ATTACHMENT_LABELS[normalized_type]
        file_name = str(attachment.get("file_name") or Path(media_path).name)
        user_content = f"已发送{label}"
        reply = f"我已接收到{label}，请给我下一步指令。"
        data = {"media_type": normalized_type, "file_name": file_name}
        for name in ("preview_path", "poster_path"):
            if attachment.get(name):
                data[name] = attachment[name]
        self.store.record_message(
            session_id,
            "user",
            user_content,
            intent="attachment_received",
            metadata={"attachment": attachment},
        )
        self.store.record_message(
            session_id,
            "assistant",
            reply,
            intent="attachment_received",
            metadata={"ok": True, "attachment_received": True, "data": data},
        )
        return {
            "ok": True,
            "session_id": session_id,
            "intent": "attachment_received",
            "confidence": 1.0,
            "recognizer_source": "attachment_handler",
            "recognition_metadata": {},
            "tool_name": "",
            "reply": reply,
            "data": data,
            "attachment": attachment,
            "attachment_received": True,
        }

    def _run_skill_plan(
        self,
        message: str,
        *,
        session_id: str,
        context: Dict[str, Any],
        planning_context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if self.skill_planner is None:
            raise SkillPlanningError("未配置 Skill Planner")
        plan = self.skill_planner.plan(
            message,
            catalog=self.skill_catalog(),
            context=planning_context,
        )
        if not isinstance(plan, SkillPlan):
            raise SkillPlanningError("Skill Planner 必须返回 SkillPlan")
        if plan.needs_clarification:
            return {
                "ok": False,
                "session_id": session_id,
                "intent": "skill_plan",
                "confidence": 1.0,
                "recognizer_source": "llm_skill_planner",
                "recognition_metadata": {"plan": plan.to_dict()},
                "tool_name": "skill_orchestrator",
                "reply": plan.clarification or "请补充完成任务所需的信息。",
                "data": {"plan": plan.to_dict(), "steps": []},
            }
        if not plan.steps:
            raise SkillPlanningError("模型没有返回可执行的 Skill 步骤")
        if len(plan.steps) > 6:
            raise SkillPlanningError("单次计划最多允许 6 个 Skill 步骤")

        self._validate_controlled_steps(message, plan)
        step_results: list[Dict[str, Any]] = []
        for index, step in enumerate(plan.steps):
            if step.skill_name not in self.skill_registry.names():
                raise SkillPlanningError(f"模型请求了未注册的 Skill：{step.skill_name}")
            arguments = self._resolve_step_references(step.arguments, step_results)
            spec = self.skill_registry.spec(step.skill_name)
            for key in spec.allowed_inputs:
                if key not in arguments and key in context:
                    arguments[key] = context[key]
            result = self.run_skill(
                step.skill_name,
                session_id=session_id,
                arguments=arguments,
            )
            step_results.append(
                {
                    "index": index,
                    "skill_name": step.skill_name,
                    "arguments": arguments,
                    **result,
                }
            )
            if not result.get("ok"):
                break

        ok = len(step_results) == len(plan.steps) and all(
            bool(item.get("ok")) for item in step_results
        )
        replies = [str(item.get("reply") or "") for item in step_results]
        reply = "\n".join(text for text in replies if text)
        requires_attachment = any(
            bool(item.get("requires_attachment")) for item in step_results
        )
        if requires_attachment:
            reply = MISSING_MEDIA_REPLY
        if not reply:
            reply = plan.summary or ("任务执行完成。" if ok else "任务执行未完成。")
        return {
            "ok": ok,
            "session_id": session_id,
            "intent": "skill_plan",
            "confidence": 1.0,
            "recognizer_source": "llm_skill_planner",
            "recognition_metadata": {"plan": plan.to_dict()},
            "tool_name": "skill_orchestrator",
            "reply": reply,
            "requires_attachment": requires_attachment,
            "data": {
                "plan": plan.to_dict(),
                "steps": step_results,
                "completed_steps": sum(bool(item.get("ok")) for item in step_results),
            },
        }

    @staticmethod
    def _resolve_step_references(
        value: Any, step_results: list[Dict[str, Any]]
    ) -> Any:
        if isinstance(value, dict):
            return {
                key: AgentService._resolve_step_references(item, step_results)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                AgentService._resolve_step_references(item, step_results)
                for item in value
            ]
        if not isinstance(value, str) or not value.startswith("$steps."):
            return value
        parts = value.split(".")
        if len(parts) < 3:
            raise SkillPlanningError(f"无效的步骤引用：{value}")
        try:
            current: Any = step_results[int(parts[1])]
            for part in parts[2:]:
                if isinstance(current, Mapping):
                    current = current[part]
                elif isinstance(current, list):
                    current = current[int(part)]
                else:
                    raise KeyError(part)
            return current
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise SkillPlanningError(f"无法解析步骤引用：{value}") from exc

    @staticmethod
    def _validate_controlled_steps(message: str, plan: SkillPlan) -> None:
        explicit_intent = RuleBasedIntentRecognizer().recognize(message).intent
        review_patterns = {
            "confirm": re.compile(r"(?:确认|认定).*(?:检测|结果|异物)"),
            "reject": re.compile(r"(?:驳回|误报|假阳性|不是异物)"),
            "close": re.compile(r"(?:闭环|处理完成|处置完成|已清除|关闭检测)"),
            "reopen": re.compile(r"(?:重开|重新打开|恢复复核)"),
        }
        for step in plan.steps:
            default_action = "confirm" if step.skill_name == "review-detection" else ""
            action = str(step.arguments.get("action") or default_action).lower()
            if step.skill_name == "control-alarm" and action in {"confirm", "cancel"}:
                expected = (
                    Intent.CONFIRM_ALARM if action == "confirm" else Intent.CANCEL_ALARM
                )
                if explicit_intent != expected:
                    raise SkillPlanningError("报警确认或取消必须来自用户的明确动作指令")
            if step.skill_name == "review-detection" and action in review_patterns:
                if not review_patterns[action].search(message):
                    raise SkillPlanningError("检测复核写操作必须来自用户的明确动作指令")

    def chat(
        self,
        message: str,
        *,
        session_id: str = "default",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        message = message.strip()
        if not message:
            raise ValueError("聊天消息不能为空")
        raw_context = dict(context or {})
        incoming_attachment = self._attachment_metadata(raw_context)
        incoming_context = {
            name: value
            for name, value in raw_context.items()
            if name != "_attachment_preview"
        }
        stored_attachment_context = (
            {} if incoming_attachment else self._latest_attachment_context(session_id)
        )
        planning_request_context = {**stored_attachment_context, **incoming_context}
        user_metadata = (
            {"attachment": incoming_attachment} if incoming_attachment else None
        )
        self.store.record_message(session_id, "user", message, metadata=user_metadata)
        recognition_context = {
            "session_id": session_id,
            "history": self.store.list_messages(session_id, limit=12),
            "request_context": planning_request_context,
        }
        match = self.recognizer.recognize(message, context=recognition_context)

        use_skill_planner = self.skill_planner is not None and match.intent != Intent.HELP and (
            self.skill_planner_mode == "always"
            or match.intent == Intent.UNKNOWN
            or bool(PLANNING_HINT.search(message))
        )
        should_use_stored_attachment = use_skill_planner or match.intent in {
            Intent.DETECT_IMAGE,
            Intent.DETECT_VIDEO,
        }
        resolved_context = dict(incoming_context)
        if should_use_stored_attachment and not incoming_attachment:
            resolved_context = {**stored_attachment_context, **incoming_context}
        if use_skill_planner:
            try:
                response = self._run_skill_plan(
                    message,
                    session_id=session_id,
                    context=resolved_context,
                    planning_context=recognition_context,
                )
            except SkillPlanningError as exc:
                response = {
                    "ok": False,
                    "session_id": session_id,
                    "intent": "skill_plan",
                    "confidence": 0.0,
                    "recognizer_source": "llm_skill_planner_error",
                    "recognition_metadata": {"error": str(exc)},
                    "tool_name": "skill_orchestrator",
                    "reply": f"大模型任务规划失败：{exc}",
                    "data": {},
                }
        elif match.intent in {Intent.HELP, Intent.UNKNOWN}:
            reply = HELP_TEXT if match.intent == Intent.HELP else f"我还不能确定你的意图。{HELP_TEXT}"
            response = {
                "ok": match.intent == Intent.HELP,
                "session_id": session_id,
                "intent": match.intent.value,
                "confidence": match.confidence,
                "recognizer_source": match.source,
                "recognition_metadata": match.metadata,
                "tool_name": "",
                "reply": reply,
                "data": {},
            }
        else:
            routed = self.router.dispatch(match, session_id, resolved_context)
            if routed.get("requires_attachment") and match.intent in {
                Intent.DETECT_IMAGE,
                Intent.DETECT_VIDEO,
            }:
                routed["reply"] = MISSING_MEDIA_REPLY
            response = {
                "session_id": session_id,
                "intent": match.intent.value,
                "confidence": match.confidence,
                "recognizer_source": match.source,
                "recognition_metadata": match.metadata,
                **routed,
            }

        if incoming_attachment:
            response["attachment_received"] = True
            response["attachment"] = incoming_attachment

        self.store.record_message(
            session_id,
            "assistant",
            str(response["reply"]),
            intent=str(response.get("intent") or match.intent.value),
            tool_name=str(response.get("tool_name") or ""),
            metadata={
                "ok": bool(response.get("ok")),
                "data": response.get("data") or {},
                "requires_attachment": bool(response.get("requires_attachment")),
                "recognizer_source": str(response.get("recognizer_source") or match.source),
                "recognition_metadata": response.get("recognition_metadata")
                or match.metadata,
            },
        )
        return response

    def history(self, session_id: str = "default", limit: int = 50) -> list[Dict[str, Any]]:
        return self.store.list_messages(session_id, limit=limit)


def create_default_service(
    db_path: Path | str | None = None,
    *,
    model_recognizer: Optional[IntentRecognizer] = None,
    recognition_mode: RecognitionMode | str = RecognitionMode.HYBRID,
    skill_planner: Optional[SkillPlanner] = None,
    skill_planner_mode: str = "hybrid",
) -> AgentService:
    store = SQLiteHistoryStore(db_path or OUTPUTS_DIR / "agent_history.sqlite3")
    return AgentService(
        store,
        model_recognizer=model_recognizer,
        recognition_mode=recognition_mode,
        skill_planner=skill_planner,
        skill_planner_mode=skill_planner_mode,
    )
