from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from project_config import OUTPUTS_DIR
from storage import SQLiteHistoryStore

from .intents import Intent
from .recognizers import (
    HybridIntentRecognizer,
    IntentRecognizer,
    RecognitionMode,
)
from .router import ToolRouter
from .tools import AgentTools


HELP_TEXT = (
    "我支持：检测这张图片、检测这段视频、查询上一轮结果、统计今天高风险报警次数、"
    "生成今日风险报告、确认报警、取消报警。"
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
    ) -> None:
        if recognizer is not None and model_recognizer is not None:
            raise ValueError("recognizer 与 model_recognizer 不能同时提供")
        self.store = store or (
            tools.store
            if tools is not None
            else SQLiteHistoryStore(OUTPUTS_DIR / "agent_history.sqlite3")
        )
        self.tools = tools or AgentTools(self.store)
        self.recognizer = recognizer or HybridIntentRecognizer(
            model_recognizer=model_recognizer,
            mode=recognition_mode,
            model_confidence_threshold=model_confidence_threshold,
        )
        self.router = ToolRouter()
        self.router.register(Intent.DETECT_IMAGE, "image_detection", self.tools.detect_image)
        self.router.register(Intent.DETECT_VIDEO, "video_detection", self.tools.detect_video)
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
        self.router.register(Intent.CONFIRM_ALARM, "alarm_control", self.tools.confirm_alarm)
        self.router.register(Intent.CANCEL_ALARM, "alarm_control", self.tools.cancel_alarm)

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
        self.store.record_message(session_id, "user", message)
        recognition_context = {
            "session_id": session_id,
            "history": self.store.list_messages(session_id, limit=12),
            "request_context": dict(context or {}),
        }
        match = self.recognizer.recognize(message, context=recognition_context)

        if match.intent in {Intent.HELP, Intent.UNKNOWN}:
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
            routed = self.router.dispatch(match, session_id, context)
            response = {
                "session_id": session_id,
                "intent": match.intent.value,
                "confidence": match.confidence,
                "recognizer_source": match.source,
                "recognition_metadata": match.metadata,
                **routed,
            }

        self.store.record_message(
            session_id,
            "assistant",
            str(response["reply"]),
            intent=match.intent.value,
            tool_name=str(response.get("tool_name") or ""),
            metadata={
                "ok": bool(response.get("ok")),
                "data": response.get("data") or {},
                "requires_attachment": bool(response.get("requires_attachment")),
                "recognizer_source": match.source,
                "recognition_metadata": match.metadata,
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
) -> AgentService:
    store = SQLiteHistoryStore(db_path or OUTPUTS_DIR / "agent_history.sqlite3")
    return AgentService(
        store,
        model_recognizer=model_recognizer,
        recognition_mode=recognition_mode,
    )
