from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from zoneinfo import ZoneInfo

from project_config import OUTPUTS_DIR, PROJECT_ROOT
from storage import SQLiteHistoryStore

from .knowledge_base import (
    DETAILED_SOURCE_PATTERN,
    KnowledgeAnswerer,
    ProjectKnowledgeBase,
)
from .intents import Intent
from .intents import RuleBasedIntentRecognizer
from .planners import SkillPlan, SkillPlanner, SkillPlanningError, SkillPlanStep
from .recognizers import (
    HybridIntentRecognizer,
    IntentRecognizer,
    RecognitionMode,
)
from .router import ToolRouter
from .skills import SkillRegistry, create_builtin_skill_registry
from .temporal import DEFAULT_TIMEZONE, resolve_temporal_expression
from .tools import AgentTools, DetectionExplainer


HELP_TEXT = (
    "我支持：检测这张图片、检测这段视频、查询上一轮结果、统计今天高风险报警次数、"
    "生成今日风险报告、确认报警、取消报警。还可通过 Skill 接口执行风险研判、"
    "按时间/风险/线路查询、人工复核和组合巡检任务。"
    "还可检查已注册固定监控源的 RTSP 连接状态。"
    "也可从固定监控源采集一个定长本地视频片段。"
    "还可对已注册 RTSP 监控执行单次采集、异物检测、风险研判和报警闭环入库。"
    "还可创建最长 24 小时的非全天候监控任务，并查询状态或人工停止。"
    "还可持续归档 RTSP 录像、保留最近 24 小时，并按绝对时间范围检测历史录像。"
)
MISSING_MEDIA_REPLY = "还没有看到图片/视频，发过来立刻帮你分析。"
ATTACHMENT_LABELS = {"image": "图片", "video": "视频"}

PLANNING_HINT = re.compile(
    r"(?:\d{1,2}[：:]\d{2}.*(?:-|到|至).*\d{1,2}[：:]\d{2}|"
    r"今天.*(?:上午|下午|早上|晚上)|"
    r"(?:凌晨|早上|上午|中午|下午|晚上)?\s*\d{1,2}(?:点|时).*?(?:到|至).*?\d{1,2}(?:点|时)|"
    r"ROI|抽帧|阈值|线路|"
    r"昨天|前天|最近\s*(?:\d+|一|两|半)*(?:分钟|小时|天)|"
    r"第?\s*\d+(?:\.\d+)?\s*分钟|从\s*\d{1,2}[：:]\d{2}\s*检测到|"
    r"风险研判|人工复核|误报|假阳性|闭环|处理完成|"
    r"开始监控|启动监控|预约监控|停止监控|监控任务|巡检任务|"
    r"历史录像|录像归档|开始录像|停止录像|保留\s*\d+\s*小时|"
    r"为什么.*风险|处置建议|同类历史|解释目标位置|"
    r"并且|然后|同时|随后|再生成|以及)",
    re.I,
)

EXPLANATION_QUESTION_PATTERNS = (
    ("risk_reason", re.compile(r"为什么.*(?:高|中|低|无)?风险|风险.*(?:原因|依据|为什么)")),
    ("action_advice", re.compile(r"(?:有什么|给出|查看|解释)?.{0,6}(?:处置|处理|应对)建议|应该怎么处理")),
    ("similar_history", re.compile(r"(?:查看|查询|找|有没有)?.{0,6}同类历史|类似.*历史")),
    ("target_position", re.compile(r"解释.*(?:目标)?位置|目标.*(?:在哪|位置|坐标)")),
)
REALTIME_TERMS = r"(?:实时巡检|持续巡检|持续实时巡检|持续连接检测|实施巡检|实时监测|持续监测)"
REALTIME_QUERY_PATTERN = re.compile(
    rf"(?:(?:查看|查询|显示|获取|看看).{{0,24}}{REALTIME_TERMS}.{{0,8}}(?:状态|情况|进度)?|"
    rf"{REALTIME_TERMS}.{{0,16}}(?:状态|情况|进度))"
)
REALTIME_STOP_PATTERN = re.compile(
    rf"(?:(?:停止|终止|结束|取消|关闭|暂停|停掉).{{0,24}}{REALTIME_TERMS}|"
    rf"{REALTIME_TERMS}.{{0,24}}(?:停止|终止|结束|取消|关闭|暂停|停掉))"
)
REALTIME_START_PATTERN = re.compile(
    rf"(?:开始|启动|开启|安排|预约|从现在开始|从.+开始).{{0,24}}{REALTIME_TERMS}"
)
REALTIME_REPORT_PATTERN = re.compile(
    rf"(?:(?:输出|生成|查看|给我|调出).{{0,24}}(?:上一轮|上一次|最近一次|刚才)?"
    rf".{{0,12}}{REALTIME_TERMS}.{{0,12}}(?:报警|预警|风险)?报告|"
    rf"(?:上一轮|上一次|最近一次|刚才)?.{{0,12}}{REALTIME_TERMS}"
    rf".{{0,12}}(?:报警|预警|风险)报告)"
)
REALTIME_ALARM_CONFIRM_BATCH_PATTERN = re.compile(
    r"(?<!已)(?:确认|保留).{0,12}(?:本轮|全部|所有|实时巡检).{0,8}(?:报警|告警)"
)
REALTIME_ALARM_CANCEL_BATCH_PATTERN = re.compile(
    r"(?<!已)(?:取消|撤销|关闭).{0,12}(?:本轮|全部|所有|实时巡检).{0,8}(?:报警|告警)"
)
REALTIME_EVENT_LATEST_PATTERN = re.compile(
    r"(?:查看|查询|显示|给我).{0,18}(?:最近一次|最新|刚才).{0,10}(?:异物|报警|预警).{0,6}(?:报告|详情)?"
)
REALTIME_EVENT_ALL_PATTERN = re.compile(
    rf"(?:查看|查询|显示).{{0,12}}(?:本次|当前).{{0,8}}{REALTIME_TERMS}.{{0,12}}(?:所有|全部).{{0,6}}(?:异物|事件|报警)"
)
REALTIME_EVENT_ACTIVE_PATTERN = re.compile(
    r"(?:查看|查询|显示).{0,12}(?:当前|仍在|正在).{0,8}(?:持续|活动|未结束).{0,6}(?:报警|异物|事件)"
)
REALTIME_EVENT_DETAIL_PATTERN = re.compile(
    r"(?:查看|查询|显示|解释).{0,8}(realtime-[a-f0-9]{12}-event-\d{4,}).{0,8}(?:详细|详情|报告)",
    re.I,
)
KNOWLEDGE_HOWTO_PATTERN = re.compile(r"(?:如何|怎么|怎样|该怎么).{0,40}(?:使用|配置|接入|启动|查询|查看|停止|检测|保存|运行)")
KNOWLEDGE_DOMAIN_PATTERN = re.compile(
    r"(?:这个系统|本系统|项目|功能|配置|参数|文件|目录|路径|源码|代码|函数|接口|数据库|Skill|RTSP|YOLO|MediaMTX|FFmpeg|"
    r"sample_fps|known_conf|conf|unknown|NMS|报警(?:规则|报告|记录|信息|数据|机制)|"
    r"告警(?:规则|报告|记录|信息|数据|机制)|历史记录|检测历史|报警历史|历史报警|历史录像|代表帧|"
    r"实时检测|周期巡检|实时巡检|持续巡检|录像归档|Web服务|connecting|reconnecting|gpu_busy)",
    re.I,
)
KNOWLEDGE_QUESTION_PATTERN = re.compile(
    r"(?:是什么|有什么|能做什么|支持什么|怎么用|如何|怎么|怎样|为什么|为何|区别|不同|"
    r"什么意思|含义|作用|用途|在哪里|在哪儿|保存在哪|放在哪|放哪儿|分别|哪些|"
    r"都是?啥|是啥|干嘛的|干什么用|咋用|咋配置|咋接入|咋启动|咋查询|咋停止|"
    r"是否缓存|会不会|吗[？?]?$)",
    re.I,
)
KNOWLEDGE_CONTEXT_PATTERN = re.compile(
    r"^(?:它|这个|这个功能|这个模式|上述功能|该功能|那它|那这个).*(?:吗|呢|如何|怎么|为什么|哪里|保存|缓存|区别)",
    re.I,
)
DYNAMIC_STATE_PATTERN = re.compile(
    r"(?:(?:当前|现在|目前|今天|上一轮|上一次|最近一次).{0,20}(?:在线|连接状态|任务状态|运行状态|"
    r"正在运行|报警数量|几次|多少|检测结果)|(?:主监控|监控源).{0,12}(?:在线吗|是否在线|连接正常吗))"
)
STATIC_STATUS_EXPLANATION_PATTERN = re.compile(
    r"(?:为什么|为何|什么意思|含义).{0,24}(?:connecting|reconnecting|gpu_busy|interrupted|failed)",
    re.I,
)
DISPLAY_ISO_TIME_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?"
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
        timezone_name: str = DEFAULT_TIMEZONE,
        knowledge_base: Optional[ProjectKnowledgeBase] = None,
        knowledge_answerer: Optional[KnowledgeAnswerer] = None,
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
        self.timezone_name = timezone_name
        self.knowledge_base = knowledge_base or ProjectKnowledgeBase(PROJECT_ROOT)
        self.knowledge_answerer = knowledge_answerer
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
            spec = self.skill_registry.spec(skill_name)
            arguments = {
                name: value
                for name, value in {**defaults, **context}.items()
                if name in spec.allowed_inputs
            }
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
        result = self.skill_registry.invoke(
            skill_name,
            session_id=session_id,
            arguments=arguments or {},
        ).to_dict()
        self._decorate_detection_data(result.get("data"), session_id=session_id)
        return self._format_output_times(result)

    def _format_output_times(self, value: Any) -> Any:
        """Format agent-facing timestamps in local time without ISO timezone suffixes."""
        if isinstance(value, dict):
            return {name: self._format_output_times(item) for name, item in value.items()}
        if isinstance(value, list):
            return [self._format_output_times(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._format_output_times(item) for item in value)
        if not isinstance(value, str):
            return value

        timezone_value = ZoneInfo(self.timezone_name)

        def replace(match: re.Match[str]) -> str:
            raw = match.group(0)
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                if parsed.tzinfo is not None:
                    parsed = parsed.astimezone(timezone_value)
                return parsed.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                return raw.replace("T", " ")

        return DISPLAY_ISO_TIME_PATTERN.sub(replace, value)

    def _present_skill_reply(self, response: Dict[str, Any]) -> None:
        """Keep deterministic Skill data intact while making its chat text operator-friendly."""
        if response.get("mode") == "knowledge":
            return
        raw = str(response.get("reply") or "").strip()
        data = response.get("data") if isinstance(response.get("data"), Mapping) else {}
        if raw.startswith(("{", "[")):
            if data.get("online") is not None:
                raw = "监控源当前在线。" if data.get("online") else "监控源当前不在线。"
            elif data.get("count") is not None:
                raw = f"查询完成，共找到 {data.get('count')} 条记录。"
            elif data.get("found") is False:
                raw = "没有找到符合条件的记录。"
            else:
                raw = "查询已完成，结果已整理展示。"
        response["reply"] = self.knowledge_base.localize_user_text(raw)

    @staticmethod
    def _explanation_question_type(message: str) -> str:
        for question_type, pattern in EXPLANATION_QUESTION_PATTERNS:
            if pattern.search(message):
                return question_type
        return ""

    @staticmethod
    def _realtime_control_action(message: str) -> str:
        if KNOWLEDGE_HOWTO_PATTERN.search(message):
            return ""
        if REALTIME_STOP_PATTERN.search(message):
            return "stop"
        if REALTIME_QUERY_PATTERN.search(message) or REALTIME_REPORT_PATTERN.search(message):
            return "query"
        return ""

    @staticmethod
    def _realtime_alarm_batch_action(message: str) -> str:
        if REALTIME_ALARM_CONFIRM_BATCH_PATTERN.search(message):
            return "confirm"
        if REALTIME_ALARM_CANCEL_BATCH_PATTERN.search(message):
            return "cancel"
        return ""

    @staticmethod
    def _realtime_event_query_arguments(message: str) -> Dict[str, Any]:
        detail = REALTIME_EVENT_DETAIL_PATTERN.search(message)
        if detail:
            return {"action": "query", "event_id": detail.group(1).lower(), "events_only": True}
        if REALTIME_EVENT_ACTIVE_PATTERN.search(message):
            return {"action": "query", "active_only": True, "events_only": True, "limit": 100}
        if REALTIME_EVENT_ALL_PATTERN.search(message):
            return {"action": "query", "events_only": True, "limit": 100}
        if REALTIME_EVENT_LATEST_PATTERN.search(message):
            return {"action": "query", "latest": True, "events_only": True, "limit": 1}
        return {}

    @staticmethod
    def _should_use_knowledge(
        message: str,
        match: Any,
        *,
        has_knowledge_context: bool,
    ) -> bool:
        if ProjectKnowledgeBase.answer_mode(message) == "developer":
            return True
        if STATIC_STATUS_EXPLANATION_PATTERN.search(message):
            return True
        if DYNAMIC_STATE_PATTERN.search(message):
            return False
        if has_knowledge_context and KNOWLEDGE_CONTEXT_PATTERN.search(message.strip()):
            return True
        if KNOWLEDGE_HOWTO_PATTERN.search(message) and KNOWLEDGE_DOMAIN_PATTERN.search(message):
            return True
        if match.intent == Intent.HELP:
            return True
        if match.intent != Intent.UNKNOWN:
            return False
        return bool(
            KNOWLEDGE_DOMAIN_PATTERN.search(message)
            and KNOWLEDGE_QUESTION_PATTERN.search(message)
        )

    def _latest_knowledge_query(self, session_id: str) -> str:
        for item in reversed(self.store.list_messages(session_id, limit=30)):
            if item.get("role") != "assistant":
                continue
            metadata = item.get("metadata")
            if not isinstance(metadata, Mapping) or metadata.get("mode") != "knowledge":
                continue
            return str(metadata.get("knowledge_query") or "").strip()
        return ""

    def _classify_ambiguous_request(
        self,
        message: str,
        *,
        history: list[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Use the LLM only as a non-authoritative router for ambiguous text."""
        classifier = getattr(self.knowledge_answerer, "classify_request", None)
        if not callable(classifier):
            return {}
        try:
            raw = classifier(message, history[-8:])
        except Exception:
            return {}
        if not isinstance(raw, Mapping):
            return {}
        mode = str(raw.get("mode") or "").strip().lower()
        aliases = {
            "command": "execute",
            "action": "execute",
            "status": "dynamic",
            "history": "dynamic",
            "question": "knowledge",
            "clarification": "clarify",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"execute", "dynamic", "knowledge", "clarify"}:
            return {}
        result = {"mode": mode}
        clarification = str(raw.get("clarification") or "").strip()
        if clarification:
            result["clarification"] = clarification[:300]
        return result

    @staticmethod
    def _knowledge_answer_is_safe(answer: str, evidence: list[Dict[str, Any]]) -> bool:
        text = answer.strip()
        if not text or len(text) > 2000:
            return False
        evidence_text = "\n".join(
            f"{item.get('source', '')}\n{item.get('excerpt', '')}" for item in evidence
        ).replace("\\", "/").lower()
        claimed_paths = re.findall(
            r"[A-Za-z0-9_./\\-]+\.(?:md|py|json|sqlite3|pt|txt|mp4)",
            text,
            re.I,
        )
        if any(path.replace("\\", "/").lower() not in evidence_text for path in claimed_paths):
            return False
        if re.search(
            r"(?:当前|现在|目前).{0,12}(?:在线|正在运行|共有\s*\d+|报警\s*\d+|任务\s*\d+)",
            text,
        ):
            return False
        return True

    def _answer_project_knowledge(
        self,
        message: str,
        *,
        previous_query: str,
        history: list[Dict[str, Any]],
    ) -> Dict[str, Any]:
        contextual = bool(previous_query and KNOWLEDGE_CONTEXT_PATTERN.search(message.strip()))
        retrieval_query = f"{previous_query} {message}" if contextual else message
        answer_mode = self.knowledge_base.answer_mode(message)
        hits = self.knowledge_base.search(retrieval_query, limit=6)
        evidence = [item.to_dict() for item in hits]
        if not hits:
            return {
                "ok": True,
                "mode": "knowledge",
                "intent": "project_knowledge",
                "confidence": 0.0,
                "recognizer_source": "knowledge_retrieval",
                "tool_name": "project_knowledge",
                "reply": self.knowledge_base.fallback_answer(
                    retrieval_query, hits, answer_mode=answer_mode
                ),
                "needs_clarification": True,
                "knowledge_query": retrieval_query,
                "data": {
                    "knowledge_sources": [],
                    "retrieval": [],
                    "answer_mode": answer_mode,
                },
            }
        answer = ""
        answer_source = "fallback"
        if self.knowledge_answerer is not None:
            try:
                candidate = str(
                    self.knowledge_answerer.answer_project_question(
                        message,
                        evidence,
                        history[-8:],
                    )
                ).strip()
                if self._knowledge_answer_is_safe(candidate, evidence):
                    answer = (
                        self.knowledge_base.present_user_answer(message, candidate)
                        if answer_mode == "user"
                        else candidate
                    )
                    answer_source = "llm"
            except Exception:
                answer = ""
        if not answer:
            answer = self.knowledge_base.fallback_answer(
                retrieval_query, hits, answer_mode=answer_mode
            )
        sources: list[Dict[str, str]] = []
        seen = set()
        for hit in hits:
            if hit.source in seen:
                continue
            seen.add(hit.source)
            sources.append({"source": hit.source, "heading": hit.heading})
            if len(sources) >= 5:
                break
        detailed_sources = bool(
            answer_mode == "developer" or DETAILED_SOURCE_PATTERN.search(message)
        )
        reference_labels: list[str] = []
        for item in sources:
            label = self.knowledge_base.source_label(
                item["source"], detailed=detailed_sources
            )
            if label not in reference_labels:
                reference_labels.append(label)
        reference_text = "、".join(reference_labels)
        return {
            "ok": True,
            "mode": "knowledge",
            "intent": "project_knowledge",
            "confidence": max((item.score for item in hits), default=0.0),
            "recognizer_source": "knowledge_retrieval",
            "tool_name": "project_knowledge",
            "reply": f"{answer}\n\n参考：{reference_text}" if reference_text else answer,
            "needs_clarification": False,
            "knowledge_query": retrieval_query,
            "data": {
                "knowledge_sources": sources,
                "retrieval": evidence,
                "answer_source": answer_source,
                "answer_mode": answer_mode,
            },
        }

    def _decorate_detection_data(
        self,
        value: Any,
        *,
        session_id: str,
        inherited_source_name: str = "",
    ) -> None:
        if isinstance(value, list):
            for item in value:
                self._decorate_detection_data(
                    item,
                    session_id=session_id,
                    inherited_source_name=inherited_source_name,
                )
            return
        if not isinstance(value, dict):
            return
        source_name = str(
            value.get("display_name")
            or value.get("monitor_source")
            or inherited_source_name
            or ""
        )
        skip_detection_presentation = bool(
            value.pop("_skip_detection_presentation", False)
        )
        detection_id = str(value.get("detection_id") or "").strip()
        is_detection_result = bool(
            detection_id
            and (
                value.get("alarm_report")
                or value.get("alarm_json")
                or (value.get("class_counts") is not None and value.get("risk_level"))
            )
        )
        if (
            is_detection_result
            and not skip_detection_presentation
            and not value.get("structured_alert")
        ):
            presentation = self.tools.detection_presentation(
                session_id,
                detection_id,
                source_name=source_name,
                representative_frames=value.get("event_frames") or [],
            )
            value.update(presentation)
        for child in list(value.values()):
            if child is value.get("structured_alert") or child is value.get("authoritative_facts"):
                continue
            if isinstance(child, (dict, list)):
                self._decorate_detection_data(
                    child,
                    session_id=session_id,
                    inherited_source_name=source_name,
                )

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
        plan = self._apply_temporal_resolution(
            plan,
            planning_context.get("temporal_resolution"),
        )
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
    def _apply_temporal_resolution(
        plan: SkillPlan,
        temporal_resolution: Any,
    ) -> SkillPlan:
        if not isinstance(temporal_resolution, Mapping):
            return plan
        kind = str(temporal_resolution.get("kind") or "")
        if kind not in {"absolute", "offset"}:
            return plan
        updated_steps = []
        for step in plan.steps:
            arguments = dict(step.arguments)
            if kind == "absolute" and step.skill_name in {
                "query-history",
                "generate-risk-report",
                "start-monitoring-task",
                "start-realtime-inspection",
                "detect-archived-video",
            }:
                arguments.pop("date", None)
                if step.skill_name in {"start-monitoring-task", "start-realtime-inspection"}:
                    arguments.pop("run_duration_seconds", None)
                arguments["start_time"] = temporal_resolution["start_time"]
                arguments["end_time"] = temporal_resolution["end_time"]
            if kind == "offset" and step.skill_name in {
                "detect-video",
                "run-inspection-task",
            }:
                arguments["start_offset_seconds"] = temporal_resolution[
                    "start_offset_seconds"
                ]
                arguments["end_offset_seconds"] = temporal_resolution[
                    "end_offset_seconds"
                ]
            updated_steps.append(SkillPlanStep(step.skill_name, arguments))
        return SkillPlan(
            steps=tuple(updated_steps),
            needs_clarification=plan.needs_clarification,
            clarification=plan.clarification,
            summary=plan.summary,
        )

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
        start_monitoring = re.compile(
            r"(?:(?:开始|启动|开启|创建|安排|预约).{0,16}(?:监控|巡检)|"
            r"(?:立即|现在)(?:开始)?(?:监控|巡检)|"
            r"(?:监控|巡检).*(?:从|到|至|持续|分钟|小时|今天|明天))"
        )
        stop_monitoring = re.compile(
            r"(?:(?:停止|终止|结束|取消|关闭).{0,16}(?:监控|巡检|任务)|"
            r"(?:监控|巡检|任务).{0,16}(?:停止|终止|结束|取消|关闭)|不再监控|别监控了)"
        )
        start_archive = re.compile(
            r"(?:(?:开始|启动|开启).{0,12}(?:录像|录制|归档)|"
            r"(?:持续|全天).{0,12}(?:录像|录制|保存录像))"
        )
        stop_archive = re.compile(
            r"(?:(?:停止|终止|结束|取消|关闭).{0,12}(?:录像|录制|归档)|"
            r"(?:录像|录制|归档).{0,12}(?:停止|终止|结束|取消|关闭))"
        )
        start_realtime = REALTIME_START_PATTERN
        stop_realtime = REALTIME_STOP_PATTERN
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
            if step.skill_name == "start-monitoring-task":
                if not start_monitoring.search(message):
                    raise SkillPlanningError("启动监控任务必须来自用户明确的开始或安排监控指令")
            if step.skill_name == "control-monitoring-task" and action == "stop":
                if not stop_monitoring.search(message):
                    raise SkillPlanningError("停止监控任务必须来自用户明确的停止指令")
            if step.skill_name == "start-realtime-inspection" and not start_realtime.search(message):
                raise SkillPlanningError("启动实时巡检必须来自用户明确的启动指令")
            if step.skill_name == "control-realtime-inspection" and action == "stop" and not stop_realtime.search(message):
                raise SkillPlanningError("停止实时巡检必须来自用户明确的停止指令")
            if step.skill_name == "control-stream-archive" and action == "start":
                if not start_archive.search(message):
                    raise SkillPlanningError("启动录像归档必须来自用户明确的开始录像指令")
            if step.skill_name == "control-stream-archive" and action == "stop":
                if not stop_archive.search(message):
                    raise SkillPlanningError("停止录像归档必须来自用户明确的停止录像指令")

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
        current_time = self.tools.current_time()
        try:
            temporal_resolution = resolve_temporal_expression(
                message,
                now=current_time,
                timezone_name=self.timezone_name,
            )
        except ValueError as exc:
            temporal_resolution = {
                "kind": "invalid",
                "error": str(exc),
            }
        localized_time = current_time.astimezone(ZoneInfo(self.timezone_name))
        recognition_context = {
            "session_id": session_id,
            "current_date": localized_time.date().isoformat(),
            "current_time": localized_time.isoformat(timespec="seconds"),
            "timezone": self.timezone_name,
            "temporal_resolution": temporal_resolution,
            "video_sources": self.tools.video_source_catalog(),
            "history": self.store.list_messages(session_id, limit=12),
            "request_context": planning_request_context,
        }
        match = self.recognizer.recognize(message, context=recognition_context)
        explanation_question_type = self._explanation_question_type(message)
        realtime_control_action = self._realtime_control_action(message)
        realtime_alarm_batch_action = self._realtime_alarm_batch_action(message)
        realtime_event_arguments = self._realtime_event_query_arguments(message)
        previous_knowledge_query = self._latest_knowledge_query(session_id)
        use_knowledge = self._should_use_knowledge(
            message,
            match,
            has_knowledge_context=bool(previous_knowledge_query),
        )
        llm_route: Dict[str, str] = {}
        if (
            not use_knowledge
            and match.intent == Intent.UNKNOWN
            and not explanation_question_type
            and not realtime_control_action
            and not realtime_alarm_batch_action
        ):
            llm_route = self._classify_ambiguous_request(
                message,
                history=recognition_context["history"],
            )
            if llm_route.get("mode") == "knowledge":
                use_knowledge = True

        has_planning_hint = bool(PLANNING_HINT.search(message))
        simple_attachment_detection = (
            match.intent in {Intent.DETECT_IMAGE, Intent.DETECT_VIDEO}
            and not has_planning_hint
        )
        use_skill_planner = (
            self.skill_planner is not None
            and not use_knowledge
            and match.intent != Intent.HELP
            and not simple_attachment_detection
            and (
                self.skill_planner_mode == "always"
                or match.intent == Intent.UNKNOWN
                or has_planning_hint
            )
        )
        should_use_stored_attachment = use_skill_planner or match.intent in {
            Intent.DETECT_IMAGE,
            Intent.DETECT_VIDEO,
        }
        resolved_context = dict(incoming_context)
        if should_use_stored_attachment and not incoming_attachment:
            resolved_context = {**stored_attachment_context, **incoming_context}
        realtime_session_id = str(
            resolved_context.get("task_session_id") or session_id
        )
        if realtime_event_arguments:
            for name in ("task_id", "source_id"):
                if resolved_context.get(name):
                    realtime_event_arguments[name] = resolved_context[name]
            routed = self.run_skill(
                "control-realtime-inspection",
                session_id=realtime_session_id,
                arguments=realtime_event_arguments,
            )
            response = {
                "session_id": session_id,
                "intent": "query_realtime_events",
                "confidence": 1.0,
                "recognizer_source": "deterministic_realtime_event_query",
                "recognition_metadata": {"arguments": realtime_event_arguments},
                "tool_name": "control-realtime-inspection",
                **routed,
            }
        elif realtime_control_action:
            realtime_arguments: Dict[str, Any] = {"action": realtime_control_action}
            if resolved_context.get("task_id"):
                realtime_arguments["task_id"] = resolved_context["task_id"]
            if resolved_context.get("source_id"):
                realtime_arguments["source_id"] = resolved_context["source_id"]
            routed = self.run_skill(
                "control-realtime-inspection",
                session_id=realtime_session_id,
                arguments=realtime_arguments,
            )
            response = {
                "session_id": session_id,
                "intent": "control_realtime_inspection",
                "confidence": 1.0,
                "recognizer_source": "deterministic_realtime_control",
                "recognition_metadata": {"action": realtime_control_action},
                "tool_name": "control-realtime-inspection",
                **routed,
            }
        elif realtime_alarm_batch_action:
            alarm_arguments: Dict[str, Any] = {
                "action": realtime_alarm_batch_action,
                "scope": "realtime_task",
            }
            if resolved_context.get("task_id"):
                alarm_arguments["task_id"] = resolved_context["task_id"]
            routed = self.run_skill(
                "control-alarm",
                session_id=realtime_session_id,
                arguments=alarm_arguments,
            )
            response = {
                "session_id": session_id,
                "intent": f"{realtime_alarm_batch_action}_realtime_alarms",
                "confidence": 1.0,
                "recognizer_source": "deterministic_realtime_alarm_control",
                "recognition_metadata": {"action": realtime_alarm_batch_action},
                "tool_name": "control-alarm",
                **routed,
            }
        elif explanation_question_type:
            explanation_arguments: Dict[str, Any] = {
                "question": message,
                "question_type": explanation_question_type,
            }
            if resolved_context.get("detection_id"):
                explanation_arguments["detection_id"] = resolved_context["detection_id"]
            routed = self.run_skill(
                "explain-detection-result",
                session_id=session_id,
                arguments=explanation_arguments,
            )
            response = {
                "session_id": session_id,
                "intent": "explain_detection_result",
                "confidence": 1.0,
                "recognizer_source": "contextual_followup",
                "recognition_metadata": {"question_type": explanation_question_type},
                "tool_name": "explain-detection-result",
                **routed,
            }
        elif use_knowledge:
            response = {
                "session_id": session_id,
                **self._answer_project_knowledge(
                    message,
                    previous_query=previous_knowledge_query,
                    history=recognition_context["history"],
                ),
            }
        elif llm_route.get("mode") == "clarify":
            response = {
                "ok": False,
                "session_id": session_id,
                "intent": "needs_clarification",
                "confidence": 1.0,
                "recognizer_source": "llm_request_classifier",
                "recognition_metadata": {"classification": llm_route},
                "tool_name": "",
                "reply": llm_route.get("clarification")
                or "请再说明一下，你是想了解项目知识、查询当前数据，还是执行一个操作？",
                "needs_clarification": True,
                "data": {},
            }
        elif use_skill_planner:
            try:
                response = self._run_skill_plan(
                    message,
                    session_id=session_id,
                    context=resolved_context,
                    planning_context=recognition_context,
                )
            except SkillPlanningError as exc:
                if str(exc) == "模型没有返回可执行的 Skill 步骤":
                    response = {
                        "session_id": session_id,
                        **self._answer_project_knowledge(
                            message,
                            previous_query=previous_knowledge_query,
                            history=recognition_context["history"],
                        ),
                    }
                    response["recognizer_source"] = "knowledge_fallback_after_empty_skill_plan"
                    response["recognition_metadata"] = {"planner_error": str(exc)}
                else:
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

        self._decorate_detection_data(response.get("data"), session_id=session_id)
        self._present_skill_reply(response)
        response = self._format_output_times(response)

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
                "mode": str(response.get("mode") or ""),
                "knowledge_query": str(response.get("knowledge_query") or ""),
            },
        )
        return response

    def history(self, session_id: str = "default", limit: int = 50) -> list[Dict[str, Any]]:
        return self._format_output_times(self.store.list_messages(session_id, limit=limit))


def create_default_service(
    db_path: Path | str | None = None,
    *,
    model_recognizer: Optional[IntentRecognizer] = None,
    recognition_mode: RecognitionMode | str = RecognitionMode.HYBRID,
    skill_planner: Optional[SkillPlanner] = None,
    skill_planner_mode: str = "hybrid",
    detection_explainer: Optional[DetectionExplainer] = None,
    knowledge_base: Optional[ProjectKnowledgeBase] = None,
    knowledge_answerer: Optional[KnowledgeAnswerer] = None,
) -> AgentService:
    store = SQLiteHistoryStore(db_path or OUTPUTS_DIR / "agent_history.sqlite3")
    tools = AgentTools(store, detection_explainer=detection_explainer)
    return AgentService(
        store,
        tools=tools,
        model_recognizer=model_recognizer,
        recognition_mode=recognition_mode,
        skill_planner=skill_planner,
        skill_planner_mode=skill_planner_mode,
        knowledge_base=knowledge_base,
        knowledge_answerer=knowledge_answerer,
    )
