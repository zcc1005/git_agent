from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Pattern, Tuple


class Intent(str, Enum):
    DETECT_IMAGE = "detect_image"
    DETECT_VIDEO = "detect_video"
    PREVIOUS_RESULT = "previous_result"
    COUNT_HIGH_RISK_TODAY = "count_high_risk_today"
    GENERATE_DAILY_REPORT = "generate_daily_report"
    CONFIRM_ALARM = "confirm_alarm"
    CANCEL_ALARM = "cancel_alarm"
    HELP = "help"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class IntentMatch:
    intent: Intent
    confidence: float
    slots: Dict[str, str] = field(default_factory=dict)
    source: str = "rules"
    metadata: Dict[str, Any] = field(default_factory=dict)


class RuleBasedIntentRecognizer:
    """Deterministic Chinese-first intent recognition for safety operations.

    Alarm actions deliberately require explicit action words.  This keeps a
    descriptive sentence such as "查询已取消报警" from being routed as a new
    cancellation command.
    """

    _RULES: Tuple[Tuple[Intent, Tuple[Pattern[str], ...]], ...] = (
        (
            Intent.CANCEL_ALARM,
            (
                re.compile(r"(?<!已)(?:取消|撤销|停止|关闭)(?:这次|当前|本次|上次|上一轮)?报警"),
                re.compile(r"(?:cancel|dismiss)\s+(?:the\s+)?alarm", re.I),
            ),
        ),
        (
            Intent.CONFIRM_ALARM,
            (
                re.compile(r"(?<!已)(?:确认|继续|保留)(?:这次|当前|本次|上次|上一轮)?报警"),
                re.compile(r"confirm\s+(?:the\s+)?alarm", re.I),
            ),
        ),
        (
            Intent.GENERATE_DAILY_REPORT,
            (
                re.compile(r"(?:生成|汇总|导出|给我)(?:一份)?(?:今天|今日|当天)(?:的)?(?:风险|报警)(?:日报|报告|汇总)"),
                re.compile(r"(?:今天|今日|当天)(?:的)?(?:风险|报警)(?:日报|报告|汇总)"),
                re.compile(r"(?:daily|today'?s)\s+(?:risk|alarm)\s+report", re.I),
            ),
        ),
        (
            Intent.COUNT_HIGH_RISK_TODAY,
            (
                re.compile(r"(?:今天|今日|当天).*(?:几次|多少次|数量|总数|统计).*(?:高风险|高危).*(?:报警|告警)?"),
                re.compile(r"(?:今天|今日|当天).*(?:高风险|高危).*(?:报警|告警).*(?:几次|多少|数量|总数)"),
                re.compile(r"how many high[- ]risk alarms today", re.I),
            ),
        ),
        (
            Intent.PREVIOUS_RESULT,
            (
                re.compile(r"(?:查询|查看|看看|显示|告诉我)?(?:上一次|上轮|上一轮|刚才|最近一次)(?:的)?(?:检测)?结果"),
                re.compile(r"(?:查询|查看|看看|显示)(?:历史|最近)(?:检测)?记录"),
                re.compile(r"(?:previous|last|latest)\s+(?:detection\s+)?result", re.I),
            ),
        ),
        (
            Intent.DETECT_IMAGE,
            (
                re.compile(r"(?:检测|识别|分析|检查|看一下).*(?:这张|这个|上传的|当前)?(?:图片|图像|照片)"),
                re.compile(r"(?:这张|这个|上传的|当前)(?:图片|图像|照片).*(?:检测|识别|分析|检查)"),
                re.compile(r"detect\s+(?:this|the|uploaded)?\s*(?:image|photo|picture)", re.I),
            ),
        ),
        (
            Intent.DETECT_VIDEO,
            (
                re.compile(r"(?:检测|识别|分析|检查|跑一下|看一下).*(?:这段|这个|上传的|当前)?视频"),
                re.compile(r"(?:这段|这个|上传的|当前)视频.*(?:检测|识别|分析|检查)"),
                re.compile(r"detect\s+(?:this|the|uploaded)?\s*video", re.I),
            ),
        ),
        (
            Intent.HELP,
            (
                re.compile(r"^(?:帮助|怎么用|你能做什么|支持什么|help)[？?。.!\s]*$", re.I),
            ),
        ),
    )

    _ALARM_ID = re.compile(r"\b(?:alarm|ALARM)[-_][A-Za-z0-9_-]+\b")

    def recognize(
        self,
        text: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> IntentMatch:
        del context
        normalized = " ".join(text.strip().split())
        if not normalized:
            return IntentMatch(Intent.UNKNOWN, 0.0)

        slots: Dict[str, str] = {}
        alarm_id = self._ALARM_ID.search(normalized)
        if alarm_id:
            slots["alarm_id"] = alarm_id.group(0)

        for intent, patterns in self._RULES:
            if self._matches_any(patterns, normalized):
                return IntentMatch(intent, 1.0, slots)
        return IntentMatch(Intent.UNKNOWN, 0.0, slots)

    @staticmethod
    def _matches_any(patterns: Iterable[Pattern[str]], text: str) -> bool:
        return any(pattern.search(text) for pattern in patterns)
