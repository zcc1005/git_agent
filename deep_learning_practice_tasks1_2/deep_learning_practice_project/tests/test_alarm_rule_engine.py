from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from task3_alarm.alarm_rule_engine import (
    _event_time_text,
    apply_alarm_rules,
    render_alarm_report,
    write_alarm_outputs,
)
from task3_alarm.unified_alarm import convert_image_detection, validate_unified_alarm


INPUT_JSON = Path("synthetic_detection.json")
CLASS_NAMES = {
    "plastic": "塑料异物",
    "stone": "石块异物",
    "metal": "金属异物",
    "wood": "木块异物",
    "unknown": "未知异物",
}


def make_image_document(
    classes: list[str],
    *,
    bbox: list[float] | None = None,
    image_names: list[str] | None = None,
    status: str = "detected",
):
    effective_bbox = bbox or [0, 0, 100, 100]
    names = image_names or ["sample.jpg"] * len(classes)
    objects = []
    for index, (class_key, image_name) in enumerate(zip(classes, names), start=1):
        objects.append(
            {
                "image": image_name,
                "class_id": index,
                "class": class_key,
                "class_name": CLASS_NAMES[class_key],
                "confidence": 0.8 + index / 100,
                "bbox_xyxy": effective_bbox,
            }
        )
    detection = {
        "status": status,
        "timestamp": "2026-07-15 12:00:00",
        "source": names[0] if names else "sample.jpg",
        "num_images": len(set(names)) if names else 0,
        "num_detections": len(objects),
        "has_foreign_object": bool(objects),
        "objects": objects,
    }
    return convert_image_detection(detection, INPUT_JSON)


class AlarmRuleEngineTests(unittest.TestCase):
    def test_video_event_report_prefers_shanghai_real_time(self) -> None:
        event = {
            "start_video_time": "00:00:15.520",
            "end_video_time": "00:00:16.250",
            "start_real_time": "2026-07-21 09:38:15.520+00:00",
            "end_real_time": "2026-07-21 09:38:16.250+00:00",
        }

        text = _event_time_text(event, "video")

        self.assertEqual(
            text,
            (
                "北京时间2026-07-21 17:38:15.520至2026-07-21 17:38:16.250"
                "（片段内时间00:00:15.520至00:00:16.250）"
            ),
        )

    def test_video_event_report_falls_back_to_relative_time(self) -> None:
        text = _event_time_text(
            {
                "start_video_time": "00:00:05.000",
                "end_video_time": "00:00:06.000",
            },
            "video",
        )

        self.assertEqual(text, "片段内时间00:00:05.000至00:00:06.000")

    def test_no_object_is_no_alarm(self) -> None:
        ruled = apply_alarm_rules(make_image_document([]))

        self.assertEqual(ruled["overall_risk"]["level"], "none")
        self.assertEqual(ruled["overall_risk"]["code"], "NO_ALARM")
        self.assertFalse(ruled["overall_risk"]["requires_stop"])
        self.assertEqual(validate_unified_alarm(ruled), [])

    def test_skipped_detection_is_not_reported_as_safe_detection(self) -> None:
        ruled = apply_alarm_rules(make_image_document([], status="skipped"))

        self.assertEqual(ruled["overall_risk"]["level"], "none")
        self.assertEqual(ruled["overall_risk"]["code"], "DETECTION_NOT_RUN")
        self.assertIn("风险未评估", ruled["overall_risk"]["reason"])
        self.assertIn("go 命令", ruled["generated_report"]["recommended_action"])

    def test_class_baselines_and_stop_action(self) -> None:
        expected = {
            "plastic": ("low", False),
            "stone": ("medium", False),
            "wood": ("medium", False),
            "unknown": ("medium", False),
            "metal": ("high", True),
        }

        for class_key, (level, requires_stop) in expected.items():
            with self.subTest(class_key=class_key):
                ruled = apply_alarm_rules(make_image_document([class_key]))
                event_risk = ruled["events"][0]["risk"]
                self.assertEqual(event_risk["level"], level)
                self.assertEqual(event_risk["requires_stop"], requires_stop)

    def test_each_escalation_condition_raises_one_level(self) -> None:
        multi = apply_alarm_rules(
            make_image_document(["plastic", "plastic", "plastic"])
        )
        large = apply_alarm_rules(
            make_image_document(["plastic"], bbox=[0, 0, 250, 250])
        )
        long_document = make_image_document(["plastic"])
        long_document["events"][0]["duration_seconds"] = 3.0
        long_event = apply_alarm_rules(long_document)

        for ruled in (multi, large, long_event):
            with self.subTest(reason=ruled["events"][0]["risk"]["reason"]):
                self.assertEqual(ruled["events"][0]["risk"]["level"], "medium")
                self.assertTrue(
                    ruled["events"][0]["risk"]["code"].startswith("ESCALATED_")
                )

    def test_multiple_escalations_are_capped_at_high(self) -> None:
        document = make_image_document(
            ["plastic", "plastic", "plastic"], bbox=[0, 0, 250, 250]
        )
        document["events"][0]["duration_seconds"] = 3.0

        ruled = apply_alarm_rules(document)

        self.assertEqual(ruled["events"][0]["risk"]["level"], "high")
        self.assertTrue(ruled["events"][0]["risk"]["requires_stop"])

    def test_overall_risk_uses_highest_event(self) -> None:
        document = make_image_document(
            ["plastic", "metal"], image_names=["low.jpg", "high.jpg"]
        )

        ruled = apply_alarm_rules(document)

        self.assertEqual(len(ruled["events"]), 2)
        self.assertEqual(
            [event["risk"]["level"] for event in ruled["events"]],
            ["low", "high"],
        )
        self.assertEqual(ruled["overall_risk"]["level"], "high")
        self.assertTrue(ruled["overall_risk"]["requires_stop"])

    def test_text_report_and_json_are_complete(self) -> None:
        document = make_image_document(["stone"])
        ruled = apply_alarm_rules(document)
        report = render_alarm_report(ruled)

        for section in (
            "一、检测来源",
            "二、报警结论",
            "三、总体风险等级",
            "四、事件详情",
            "五、风险说明",
            "六、处理建议",
            "七、生成信息",
        ):
            self.assertIn(section, report)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_json = Path(temp_dir) / "unified_alarm.json"
            output_txt = Path(temp_dir) / "alarm_report.txt"
            written, written_report = write_alarm_outputs(
                document, output_json, output_txt
            )
            self.assertTrue(output_json.exists())
            self.assertTrue(output_txt.exists())
            self.assertEqual(written["overall_risk"]["level"], "medium")
            self.assertEqual(output_txt.read_text(encoding="utf-8"), written_report)


if __name__ == "__main__":
    unittest.main()
