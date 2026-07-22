from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from task3_alarm.unified_alarm import (  # noqa: E402
    assert_valid_unified_alarm,
    convert_detection,
)


RULE_ENGINE_NAME = "task3_alarm.alarm_rule_engine/1.0"
RISK_LEVEL_NAMES = {
    "none": "无报警",
    "low": "低风险",
    "medium": "中风险",
    "high": "高风险",
}
RISK_SCORES = {"none": 0, "low": 1, "medium": 2, "high": 3}
SCORE_LEVELS = {value: key for key, value in RISK_SCORES.items()}
REPORT_TIMEZONE = ZoneInfo("Asia/Shanghai")


# 规则基线沿用现有报警文本中的类别风险含义。当前YOLO只输出前五类，
# 其余类别保留给以后扩展数据集，避免再次修改规则引擎结构。
CLASS_RULES: Dict[str, Dict[str, Any]] = {
    "metal": {
        "base_level": "high",
        "code": "METAL_DAMAGE_RISK",
        "reason": "金属异物可能划伤皮带、损坏滚筒或引发设备故障。",
    },
    "stone": {
        "base_level": "medium",
        "code": "STONE_BLOCKING_RISK",
        "reason": "石块异物可能造成皮带跑偏、卡堵或异常冲击。",
    },
    "wood": {
        "base_level": "medium",
        "code": "WOOD_BLOCKING_RISK",
        "reason": "木块异物可能造成堵塞、影响下料或卡住输送机构。",
    },
    "plastic": {
        "base_level": "low",
        "code": "PLASTIC_QUALITY_RISK",
        "reason": "塑料异物可能影响物料分选质量和后续工序。",
    },
    "unknown": {
        "base_level": "medium",
        "code": "UNKNOWN_REVIEW_REQUIRED",
        "reason": "未知异物需要人工复核，确认其材质和设备影响。",
    },
    "large_lump": {
        "base_level": "high",
        "code": "LARGE_LUMP_IMPACT_RISK",
        "reason": "大块异物可能造成堵料、冲击设备或异常停机。",
    },
    "wire": {
        "base_level": "high",
        "code": "WIRE_ENTANGLEMENT_RISK",
        "reason": "线状异物可能缠绕滚筒或输送机构并引发停机。",
    },
    "cloth": {
        "base_level": "medium",
        "code": "CLOTH_ENTANGLEMENT_RISK",
        "reason": "布条异物可能缠绕托辊、滚筒或其他转动部件。",
    },
    "rubber": {
        "base_level": "medium",
        "code": "RUBBER_TRANSPORT_RISK",
        "reason": "橡胶异物可能影响物料输送和设备稳定运行。",
    },
    "bottle": {
        "base_level": "medium",
        "code": "BOTTLE_TRANSPORT_RISK",
        "reason": "瓶类异物可能滚动、卡堵或影响输送稳定性。",
    },
}


ACTION_RULES = {
    "none": {
        "action_code": "CONTINUE_MONITORING",
        "requires_stop": False,
        "text": "未检测到确认异物，保持正常运行并继续监测。",
    },
    "low": {
        "action_code": "MONITOR_AND_RECORD",
        "requires_stop": False,
        "text": "暂不需要停机，持续观察并保留报警记录，必要时进行人工确认。",
    },
    "medium": {
        "action_code": "SLOW_AND_INSPECT",
        "requires_stop": False,
        "text": "建议降低皮带速度并保持报警提示，通知现场人员复核并及时清理异物。",
    },
    "high": {
        "action_code": "STOP_AND_INSPECT",
        "requires_stop": True,
        "text": "立即触发声光报警并停止皮带，通知现场人员清理异物；确认皮带、托辊和滚筒无异常后再恢复运行。",
    },
}


MULTI_OBJECT_ESCALATION_COUNT = 3
LARGE_AREA_THRESHOLD = 50_000.0
LONG_DURATION_SECONDS = 3.0


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _class_rule(class_key: str) -> Dict[str, Any]:
    return CLASS_RULES.get(class_key.strip().lower(), CLASS_RULES["unknown"])


def _event_unique_count(event: Dict[str, Any]) -> int:
    summary = event.get("detection_summary") or {}
    return max(
        0,
        _integer(
            summary.get("unique_object_count"),
            len(event.get("objects") or []),
        ),
    )


def _event_peak_count(event: Dict[str, Any]) -> int:
    summary = event.get("detection_summary") or {}
    return max(
        0,
        _integer(
            summary.get("reported_peak_box_count"),
            len(event.get("objects") or []),
        ),
    )


def evaluate_event_risk(event: Dict[str, Any]) -> Dict[str, Any]:
    objects = [item for item in event.get("objects", []) if isinstance(item, dict)]
    if not objects:
        action = ACTION_RULES["none"]
        return {
            "status": "completed",
            "level": "none",
            "code": "NO_CONFIRMED_OBJECT",
            "reason": "该事件没有确认异物目标。",
            "requires_stop": action["requires_stop"],
            "action_code": action["action_code"],
        }

    object_rules = [
        (item, _class_rule(str(item.get("class") or "unknown"))) for item in objects
    ]
    _, primary_rule = max(
        object_rules,
        key=lambda pair: (
            RISK_SCORES[pair[1]["base_level"]],
            _number(pair[0].get("confidence")),
        ),
    )
    score = RISK_SCORES[primary_rule["base_level"]]
    escalation_reasons: List[str] = []

    unique_count = _event_unique_count(event)
    peak_count = _event_peak_count(event)
    if max(unique_count, peak_count) >= MULTI_OBJECT_ESCALATION_COUNT:
        score += 1
        escalation_reasons.append(
            f"确认目标或单帧同时目标达到{MULTI_OBJECT_ESCALATION_COUNT}个及以上"
        )

    max_area = max((_number(item.get("area")) for item in objects), default=0.0)
    if max_area >= LARGE_AREA_THRESHOLD:
        score += 1
        escalation_reasons.append(
            f"最大检测框面积达到{max_area:.0f}像素，超过大目标阈值{LARGE_AREA_THRESHOLD:.0f}像素"
        )

    duration = _number(event.get("duration_seconds"), 0.0)
    if duration >= LONG_DURATION_SECONDS:
        score += 1
        escalation_reasons.append(
            f"事件持续{duration:.2f}秒，达到持续时间阈值{LONG_DURATION_SECONDS:.1f}秒"
        )

    level = SCORE_LEVELS[min(3, max(1, score))]
    action = ACTION_RULES[level]
    class_names = list(
        dict.fromkeys(str(item.get("class_name") or "未知异物") for item in objects)
    )
    reason_parts = [
        f"事件包含{'、'.join(class_names)}。",
        str(primary_rule["reason"]),
    ]
    if escalation_reasons:
        reason_parts.append("风险升级原因：" + "；".join(escalation_reasons) + "。")

    code = str(primary_rule["code"])
    if level != primary_rule["base_level"]:
        code = f"ESCALATED_{code}"
    return {
        "status": "completed",
        "level": level,
        "code": code,
        "reason": "".join(reason_parts),
        "requires_stop": action["requires_stop"],
        "action_code": action["action_code"],
    }


def _overall_risk(
    events: Sequence[Dict[str, Any]], detection_status: str = ""
) -> Dict[str, Any]:
    if detection_status.strip().lower() == "skipped":
        action = ACTION_RULES["none"]
        return {
            "status": "completed",
            "level": "none",
            "code": "DETECTION_NOT_RUN",
            "reason": "当前命令未启动异物检测，因此本次风险未评估，不能据此认定现场无异物。",
            "requires_stop": action["requires_stop"],
            "action_code": action["action_code"],
        }
    if not events:
        action = ACTION_RULES["none"]
        return {
            "status": "completed",
            "level": "none",
            "code": "NO_ALARM",
            "reason": "本次检测未发现确认异物，不触发报警。",
            "requires_stop": action["requires_stop"],
            "action_code": action["action_code"],
        }

    highest_score = max(
        RISK_SCORES.get(str(event["risk"].get("level")), 0) for event in events
    )
    level = SCORE_LEVELS[highest_score]
    highest_events = [
        str(event.get("event_id"))
        for event in events
        if RISK_SCORES.get(str(event["risk"].get("level")), 0) == highest_score
    ]
    action = ACTION_RULES[level]
    return {
        "status": "completed",
        "level": level,
        "code": f"OVERALL_{level.upper()}_RISK",
        "reason": (
            f"共评估{len(events)}个确认事件，事件{','.join(highest_events)}达到"
            f"{RISK_LEVEL_NAMES[level]}，总体风险按最高事件确定。"
        ),
        "requires_stop": action["requires_stop"],
        "action_code": action["action_code"],
    }


def _format_report_real_time(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(REPORT_TIMEZONE)
    formatted = parsed.strftime("%Y-%m-%d %H:%M:%S")
    if parsed.microsecond:
        formatted += f".{parsed.microsecond // 1000:03d}"
    return formatted


def _event_time_text(event: Dict[str, Any], source_type: str) -> str:
    if source_type == "video":
        real_start = _format_report_real_time(event.get("start_real_time"))
        real_end = _format_report_real_time(event.get("end_real_time")) or real_start
        video_start = str(event.get("start_video_time") or "").strip()
        video_end = str(event.get("end_video_time") or video_start).strip()
        if real_start:
            relative_text = (
                f"（片段内时间{video_start}至{video_end}）"
                if video_start
                else ""
            )
            return f"北京时间{real_start}至{real_end}{relative_text}"
        start = video_start or "未知时间"
        end = video_end or start
        return f"片段内时间{start}至{end}"
    return str(event.get("start_real_time") or "当前检测图片")


def _report_fields(document: Dict[str, Any]) -> Dict[str, Any]:
    events = document.get("events") or []
    overall = document["overall_risk"]
    source_type = str(document.get("source", {}).get("type") or "image")
    detection_status = str(document.get("detection_summary", {}).get("status") or "")
    if detection_status.strip().lower() == "skipped":
        conclusion = "当前命令未启动异物检测，本次不生成现场风险结论。"
        explanation = str(overall["reason"])
    elif not events:
        conclusion = "本次检测未发现确认异物，不触发工业皮带异物报警。"
        explanation = str(overall["reason"])
    else:
        unique_count = sum(_event_unique_count(event) for event in events)
        conclusion = (
            f"本次检测到{len(events)}个确认异物事件，按事件统计共"
            f"{unique_count}个独立或代表目标，总体为{RISK_LEVEL_NAMES[overall['level']]}。"
        )
        details = []
        for event in events:
            class_counts = event.get("detection_summary", {}).get("class_counts") or {}
            class_text = "、".join(
                f"{name}{count}个" for name, count in class_counts.items()
            ) or "确认异物"
            details.append(
                f"事件{event['event_id']}（{_event_time_text(event, source_type)}）"
                f"检测到{class_text}，判定为{RISK_LEVEL_NAMES[event['risk']['level']]}："
                f"{event['risk']['reason']}"
            )
        explanation = "\n".join(details)
    action = ACTION_RULES[str(overall["level"])]
    recommended_action = str(action["text"])
    if detection_status.strip().lower() == "skipped":
        recommended_action = "如需评估现场风险，请输入或识别 go 命令后重新运行检测。"
    return {
        "status": "completed",
        "conclusion": conclusion,
        "risk_explanation": explanation,
        "recommended_action": recommended_action,
        "generator": RULE_ENGINE_NAME,
    }


def apply_alarm_rules(document: Dict[str, Any]) -> Dict[str, Any]:
    assert_valid_unified_alarm(document)
    ruled = deepcopy(document)
    for event in ruled.get("events", []):
        event["risk"] = evaluate_event_risk(event)
    ruled["overall_risk"] = _overall_risk(
        ruled.get("events", []),
        str(ruled.get("detection_summary", {}).get("status") or ""),
    )
    ruled["generated_report"] = _report_fields(ruled)
    assert_valid_unified_alarm(ruled)
    return ruled


def render_alarm_report(document: Dict[str, Any]) -> str:
    assert_valid_unified_alarm(document)
    overall = document["overall_risk"]
    generated = document["generated_report"]
    source = document.get("source") or {}
    events = document.get("events") or []
    lines = [
        "工业皮带异物报警报告",
        "",
        "一、检测来源",
        f"来源类型：{'视频' if source.get('type') == 'video' else '图片'}",
        f"来源文件：{source.get('name') or source.get('path') or '未知'}",
        "",
        "二、报警结论",
        str(generated.get("conclusion") or "未生成结论。"),
        "",
        "三、总体风险等级",
        f"{RISK_LEVEL_NAMES.get(str(overall.get('level')), str(overall.get('level')))}",
        f"规则代码：{overall.get('code')}",
        f"是否要求停机：{'是' if overall.get('requires_stop') else '否'}",
        "",
        "四、事件详情",
    ]
    if not events:
        lines.append("无确认异物事件。")
    else:
        for event in events:
            summary = event.get("detection_summary") or {}
            counts = summary.get("class_counts") or {}
            count_text = "、".join(f"{name}{count}个" for name, count in counts.items()) or "确认异物"
            lines.extend(
                [
                    f"事件{event.get('event_id')}：{_event_time_text(event, str(source.get('type')))}",
                    f"目标信息：{count_text}；独立或代表目标{_event_unique_count(event)}个；"
                    f"最高置信度{_number(summary.get('max_confidence')):.4f}。",
                    f"风险等级：{RISK_LEVEL_NAMES.get(str(event['risk'].get('level')), event['risk'].get('level'))}",
                    f"风险说明：{event['risk'].get('reason')}",
                    "",
                ]
            )
    lines.extend(
        [
            "五、风险说明",
            str(generated.get("risk_explanation") or overall.get("reason") or "无"),
            "",
            "六、处理建议",
            str(generated.get("recommended_action") or "继续监测。"),
            "",
            "七、生成信息",
            f"规则引擎：{generated.get('generator') or RULE_ENGINE_NAME}",
            "本报告由确定性规则生成；风险等级和处置动作未交由语言模型自由判断。",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def write_alarm_outputs(
    document: Dict[str, Any], output_json: Path, output_txt: Path
) -> Tuple[Dict[str, Any], str]:
    ruled = apply_alarm_rules(document)
    report = render_alarm_report(ruled)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(ruled, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    output_txt.write_text(report, encoding="utf-8")
    return ruled, report


def complete_detection_alarm(
    detection: Dict[str, Any],
    input_json: Path,
    output_json: Path,
    output_txt: Path,
    source_type: str = "auto",
) -> Tuple[Dict[str, Any], str]:
    unified = convert_detection(
        detection, input_json=input_json, source_type=source_type
    )
    return write_alarm_outputs(unified, output_json, output_txt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将图片或视频检测结果转换为统一报警结构并执行确定性风险规则"
    )
    parser.add_argument("--input", type=Path, required=True, help="检测结果 JSON")
    parser.add_argument("--output-json", type=Path, required=True, help="规则完成后的统一报警 JSON")
    parser.add_argument("--output-txt", type=Path, required=True, help="报警文本报告")
    parser.add_argument(
        "--source-type", choices=["auto", "image", "video"], default="auto"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        detection = json.loads(args.input.read_text(encoding="utf-8"))
        ruled, _ = complete_detection_alarm(
            detection,
            input_json=args.input,
            output_json=args.output_json,
            output_txt=args.output_txt,
            source_type=args.source_type,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(exc)
        raise SystemExit(1) from exc
    print("统一报警结构、风险规则和文本报告生成成功")
    print(f"总体风险：{RISK_LEVEL_NAMES[ruled['overall_risk']['level']]}")
    print(f"统一报警JSON：{args.output_json.resolve()}")
    print(f"报警报告：{args.output_txt.resolve()}")


if __name__ == "__main__":
    main()
