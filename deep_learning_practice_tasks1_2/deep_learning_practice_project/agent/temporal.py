from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo


DEFAULT_TIMEZONE = "Asia/Shanghai"

_NUMBER_WORDS = {
    "半": 0.5,
    "一": 1.0,
    "两": 2.0,
    "二": 2.0,
    "三": 3.0,
    "四": 4.0,
    "五": 5.0,
    "六": 6.0,
    "七": 7.0,
    "八": 8.0,
    "九": 9.0,
    "十": 10.0,
}
_RECENT_RE = re.compile(
    r"最近\s*(?P<amount>\d+(?:\.\d+)?|半|一|两|二|三|四|五|六|七|八|九|十)\s*"
    r"(?P<unit>分钟|小时|天)"
)
_MINUTE_OFFSET_RE = re.compile(
    r"第?\s*(?P<start>\d+(?:\.\d+)?)\s*分(?:钟)?\s*"
    r"(?:到|至|[-—~～])\s*第?\s*(?P<end>\d+(?:\.\d+)?)\s*分(?:钟)?"
)
_CLOCK_OFFSET_RE = re.compile(
    r"从\s*(?P<start>\d{1,2}[:：]\d{2}(?:[:：]\d{2})?)\s*检测\s*"
    r"(?:到|至)\s*(?P<end>\d{1,2}[:：]\d{2}(?:[:：]\d{2})?)"
)
_VIDEO_CLOCK_OFFSET_RE = re.compile(
    r"(?:视频|录像).*?(?P<start>\d{1,2}[:：]\d{2}(?:[:：]\d{2})?)\s*"
    r"(?:到|至|[-—~～])\s*(?P<end>\d{1,2}[:：]\d{2}(?:[:：]\d{2})?)"
)
_ISO_DATE_RE = re.compile(r"(?<!\d)(?P<date>\d{4}-\d{2}-\d{2})(?!\d)")


def _time_point(prefix: str) -> str:
    return (
        rf"(?:(?P<{prefix}_period>凌晨|早上|上午|中午|下午|晚上)\s*)?"
        rf"(?P<{prefix}_hour>\d{{1,2}})"
        rf"(?:(?:[:：](?P<{prefix}_colon_minute>\d{{1,2}}))|"
        rf"(?:(?:点|时)(?:(?P<{prefix}_minute>\d{{1,2}})分?)?))"
    )


_TIME_RANGE_RE = re.compile(
    _time_point("start")
    + r"\s*(?:到|至|[-—~～])\s*"
    + _time_point("end")
)

_PERIOD_RANGES = {
    "凌晨": (0, 6),
    "早上": (6, 9),
    "上午": (0, 12),
    "中午": (11, 14),
    "下午": (12, 18),
    "晚上": (18, 24),
}


def _iso(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _seconds_value(value: float) -> int | float:
    rounded = round(value, 6)
    return int(rounded) if rounded.is_integer() else rounded


def _clock_offset_seconds(value: str) -> float:
    parts = [int(item) for item in value.replace("：", ":").split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        if seconds >= 60:
            raise ValueError("视频偏移的秒数必须小于 60")
        return float(minutes * 60 + seconds)
    hours, minutes, seconds = parts
    if minutes >= 60 or seconds >= 60:
        raise ValueError("视频偏移必须使用 HH:MM:SS 或 MM:SS")
    return float(hours * 3600 + minutes * 60 + seconds)


def _offset_result(start: float, end: float, expression: str) -> Dict[str, Any]:
    if start < 0 or end <= start:
        raise ValueError("视频结束偏移必须大于开始偏移")
    return {
        "kind": "offset",
        "start_offset_seconds": _seconds_value(start),
        "end_offset_seconds": _seconds_value(end),
        "matched_expression": expression,
    }


def _target_date(message: str, current_date: date) -> date:
    explicit = _ISO_DATE_RE.search(message)
    if explicit:
        return date.fromisoformat(explicit.group("date"))
    if "昨天" in message:
        return current_date - timedelta(days=1)
    if "前天" in message:
        return current_date - timedelta(days=2)
    if "明天" in message:
        return current_date + timedelta(days=1)
    return current_date


def _hour_for_period(hour: int, period: str) -> int:
    if not 0 <= hour <= 23:
        raise ValueError("小时必须位于 0 到 23")
    if period in {"下午", "晚上", "中午"} and hour < 12:
        return hour + 12
    if period in {"凌晨", "早上", "上午"} and hour == 12:
        return 0
    return hour


def _point_from_match(match: re.Match[str], prefix: str, inherited_period: str = "") -> tuple[int, int, str]:
    period = str(match.group(f"{prefix}_period") or inherited_period)
    hour = _hour_for_period(int(match.group(f"{prefix}_hour")), period)
    minute_text = match.group(f"{prefix}_colon_minute") or match.group(f"{prefix}_minute") or "0"
    minute = int(minute_text)
    if not 0 <= minute <= 59:
        raise ValueError("分钟必须位于 0 到 59")
    return hour, minute, period


def resolve_temporal_expression(
    message: str,
    *,
    now: Optional[datetime] = None,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> Dict[str, Any]:
    """Resolve supported Chinese time expressions into a deterministic closed contract."""
    text = message.strip()
    if not text:
        return {}
    timezone = ZoneInfo(timezone_name)
    current = now or datetime.now(tz=timezone)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone)
    else:
        current = current.astimezone(timezone)

    minute_offsets = _MINUTE_OFFSET_RE.search(text)
    if minute_offsets:
        return _offset_result(
            float(minute_offsets.group("start")) * 60,
            float(minute_offsets.group("end")) * 60,
            minute_offsets.group(0),
        )
    clock_offsets = _CLOCK_OFFSET_RE.search(text) or _VIDEO_CLOCK_OFFSET_RE.search(text)
    if clock_offsets:
        return _offset_result(
            _clock_offset_seconds(clock_offsets.group("start")),
            _clock_offset_seconds(clock_offsets.group("end")),
            clock_offsets.group(0),
        )

    recent = _RECENT_RE.search(text)
    if recent:
        raw_amount = recent.group("amount")
        amount = _NUMBER_WORDS.get(raw_amount, float(raw_amount) if raw_amount[0].isdigit() else 0.0)
        unit = recent.group("unit")
        delta = {
            "分钟": timedelta(minutes=amount),
            "小时": timedelta(hours=amount),
            "天": timedelta(days=amount),
        }[unit]
        return {
            "kind": "absolute",
            "start_time": _iso(current - delta),
            "end_time": _iso(current),
            "matched_expression": recent.group(0),
        }

    target_day = _target_date(text, current.date())
    time_range = _TIME_RANGE_RE.search(text)
    if time_range:
        start_hour, start_minute, start_period = _point_from_match(time_range, "start")
        end_hour, end_minute, _ = _point_from_match(
            time_range,
            "end",
            inherited_period=start_period,
        )
        start = datetime.combine(
            target_day,
            time(start_hour, start_minute),
            tzinfo=timezone,
        )
        end = datetime.combine(
            target_day,
            time(end_hour, end_minute),
            tzinfo=timezone,
        )
        if end <= start:
            end += timedelta(days=1)
        return {
            "kind": "absolute",
            "start_time": _iso(start),
            "end_time": _iso(end),
            "matched_expression": time_range.group(0),
        }

    for period, (start_hour, end_hour) in _PERIOD_RANGES.items():
        if period in text and any(marker in text for marker in ("今天", "昨天", "前天", "明天")):
            start = datetime.combine(target_day, time(start_hour), tzinfo=timezone)
            end = datetime.combine(target_day, time.max, tzinfo=timezone)
            if end_hour < 24:
                end = datetime.combine(target_day, time(end_hour), tzinfo=timezone) - timedelta(seconds=1)
            return {
                "kind": "absolute",
                "start_time": _iso(start),
                "end_time": _iso(end),
                "matched_expression": f"{target_day.isoformat()} {period}",
            }

    if any(marker in text for marker in ("今天", "昨天", "前天", "明天")) or _ISO_DATE_RE.search(text):
        start = datetime.combine(target_day, time.min, tzinfo=timezone)
        end = datetime.combine(target_day, time.max, tzinfo=timezone)
        return {
            "kind": "absolute",
            "start_time": _iso(start),
            "end_time": _iso(end),
            "matched_expression": target_day.isoformat(),
        }
    return {}
