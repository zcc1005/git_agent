from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from agent import (
    AgentService,
    AgentTools,
    HybridIntentRecognizer,
    ImageDetectionOutcome,
    Intent,
    IntentMatch,
    RecognitionMode,
    RuleBasedIntentRecognizer,
    VideoDetectionOutcome,
)
from storage import SQLiteHistoryStore


FIXED_NOW = datetime(2026, 7, 16, 10, 30, tzinfo=timezone.utc)


def fake_detection_runner(
    video_path: Path, video_start: datetime, parameters: dict
) -> VideoDetectionOutcome:
    detection = {
        "status": "completed",
        "video": str(video_path),
        "num_events": 2,
        "events": [
            {"event_id": 1, "key_frame": "outputs/frames/event_1.jpg"},
            {"event_id": 2, "key_frame": "outputs/frames/event_2.jpg"},
        ],
        "class_counts": {"metal": 1, "stone": 1},
    }
    alarm = {
        "report_id": "alarm-test-001",
        "overall_risk": {"level": "high", "requires_stop": True},
    }
    return VideoDetectionOutcome(detection, alarm, "测试报警报告")


def fake_image_detection_runner(
    image_path: Path, parameters: dict
) -> ImageDetectionOutcome:
    detection = {
        "status": "detected",
        "source": str(image_path),
        "num_images": 1,
        "num_detections": 2,
        "num_candidates": 1,
        "has_foreign_object": True,
        "class_counts": {"metal": 1, "stone": 1},
        "candidate_counts": {"wood": 1},
        "objects": [{"class": "metal"}, {"class": "stone"}],
    }
    alarm = {
        "report_id": "alarm-image-test-001",
        "overall_risk": {"level": "high", "requires_stop": True},
    }
    return ImageDetectionOutcome(
        detection,
        alarm,
        "测试图片报警报告",
        visualization_image="outputs/frames/image_event_1.jpg",
    )


class StaticModelRecognizer:
    def __init__(self, match: IntentMatch) -> None:
        self.match = match
        self.calls = []

    def recognize(self, text: str, *, context=None) -> IntentMatch:
        self.calls.append({"text": text, "context": context})
        return self.match


class IntentRecognizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recognizer = RuleBasedIntentRecognizer()

    def test_required_phrases_map_to_expected_intents(self) -> None:
        cases = {
            "检测这张图片": Intent.DETECT_IMAGE,
            "检测这段视频": Intent.DETECT_VIDEO,
            "查询上一轮结果": Intent.PREVIOUS_RESULT,
            "今天有几次高风险报警": Intent.COUNT_HIGH_RISK_TODAY,
            "生成今日风险报告": Intent.GENERATE_DAILY_REPORT,
            "查看当前报警": Intent.CURRENT_ALARM,
            "确认报警": Intent.CONFIRM_ALARM,
            "取消报警": Intent.CANCEL_ALARM,
        }
        for text, intent in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self.recognizer.recognize(text).intent, intent)

    def test_short_attachment_detection_phrases_are_recognized(self) -> None:
        cases = {
            "检测图片": Intent.DETECT_IMAGE,
            "检测这个图片": Intent.DETECT_IMAGE,
            "分析图片": Intent.DETECT_IMAGE,
            "检测视频": Intent.DETECT_VIDEO,
            "检测这个视频": Intent.DETECT_VIDEO,
            "分析视频": Intent.DETECT_VIDEO,
        }
        for text, intent in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self.recognizer.recognize(text).intent, intent)

    def test_unknown_text_is_not_routed_to_alarm_action(self) -> None:
        self.assertEqual(self.recognizer.recognize("查询已取消报警").intent, Intent.UNKNOWN)


class HybridIntentRecognizerTests(unittest.TestCase):
    def test_rules_take_priority_without_calling_model(self) -> None:
        model = StaticModelRecognizer(IntentMatch(Intent.UNKNOWN, 0.9, source="model"))
        recognizer = HybridIntentRecognizer(model_recognizer=model)

        match = recognizer.recognize("查询上一轮结果")

        self.assertEqual(match.intent, Intent.PREVIOUS_RESULT)
        self.assertEqual(match.source, "hybrid_rules")
        self.assertEqual(model.calls, [])

    def test_model_is_used_only_as_unknown_intent_fallback(self) -> None:
        model = StaticModelRecognizer(
            IntentMatch(Intent.GENERATE_DAILY_REPORT, 0.92, source="model")
        )
        recognizer = HybridIntentRecognizer(model_recognizer=model)

        match = recognizer.recognize(
            "给我来份本日态势简报", context={"session_id": "s1"}
        )

        self.assertEqual(match.intent, Intent.GENERATE_DAILY_REPORT)
        self.assertEqual(match.source, "hybrid_model")
        self.assertEqual(model.calls[0]["context"]["session_id"], "s1")

    def test_model_cannot_infer_alarm_control_from_ambiguous_text(self) -> None:
        model = StaticModelRecognizer(
            IntentMatch(Intent.CANCEL_ALARM, 0.98, source="model")
        )
        recognizer = HybridIntentRecognizer(model_recognizer=model)

        match = recognizer.recognize("把它处理掉")

        self.assertEqual(match.intent, Intent.UNKNOWN)
        self.assertEqual(match.source, "hybrid_safety_guard")
        self.assertEqual(match.metadata["blocked_model_action"], "cancel_alarm")

    def test_model_mode_requires_a_model_implementation(self) -> None:
        with self.assertRaises(ValueError):
            HybridIntentRecognizer(mode=RecognitionMode.MODEL)


class AgentServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = SQLiteHistoryStore(
            Path(self.temp_dir.name) / "history.sqlite3", now=lambda: FIXED_NOW
        )
        self.control_calls = []
        tools = AgentTools(
            self.store,
            detection_runner=fake_detection_runner,
            image_detection_runner=fake_image_detection_runner,
            alarm_control_handler=lambda action, alarm: self.control_calls.append(
                (action, alarm.id)
            ),
            now=lambda: FIXED_NOW,
        )
        self.service = AgentService(self.store, tools=tools)
        self.video = Path(self.temp_dir.name) / "belt.mp4"
        self.video.write_bytes(b"fake-video")
        self.image = Path(self.temp_dir.name) / "belt.jpg"
        self.image.write_bytes(b"fake-image")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_realtime_task_can_be_queried_from_another_chat_session(self) -> None:
        with patch.object(
            self.service,
            "run_skill",
            return_value={"ok": True, "reply": "运行中。", "data": {"found": True}},
        ) as run_skill:
            result = self.service.chat(
                "查看当前实时巡检状态",
                session_id="chat-new",
                context={
                    "task_id": "realtime-task-1",
                    "task_session_id": "chat-owner",
                },
            )

        self.assertTrue(result["ok"])
        run_skill.assert_called_once()
        self.assertEqual(run_skill.call_args.kwargs["session_id"], "chat-owner")
        self.assertEqual(
            run_skill.call_args.kwargs["arguments"]["task_id"], "realtime-task-1"
        )

    def test_detection_then_history_and_risk_queries(self) -> None:
        detected = self.service.chat(
            "检测这段视频",
            session_id="s1",
            context={"video_path": str(self.video)},
        )
        previous = self.service.chat("查询上一轮结果", session_id="s1")
        count = self.service.chat("今天有几次高风险报警", session_id="s1")
        report = self.service.chat("生成今日风险报告", session_id="s1")

        self.assertTrue(detected["ok"])
        self.assertNotIn("报警编号", detected["reply"])
        self.assertEqual(detected["data"]["event_count"], 2)
        self.assertEqual(detected["data"]["alarm_report"], "测试报警报告")
        self.assertEqual(
            detected["data"]["event_frames"][0]["key_frame"],
            "outputs/frames/event_1.jpg",
        )
        self.assertTrue(previous["data"]["found"])
        self.assertEqual(previous["data"]["alarm_report"], "测试报警报告")
        self.assertEqual(len(previous["data"]["event_frames"]), 2)
        self.assertEqual(previous["data"]["risk_level"], "high")
        self.assertEqual(count["data"]["high_risk_count"], 1)
        self.assertEqual(report["data"]["detection_count"], 1)
        self.assertIn("高/中/低风险：1/0/0", report["reply"])

    def test_detection_without_video_requests_attachment(self) -> None:
        response = self.service.chat("检测这段视频", session_id="s1")
        self.assertFalse(response["ok"])
        self.assertTrue(response["requires_attachment"])
        self.assertEqual(response["reply"], "还没有看到图片/视频，发过来立刻帮你分析。")

    def test_attachment_only_turn_is_reused_by_next_instruction(self) -> None:
        received = self.service.receive_attachment(
            "video",
            self.video,
            session_id="pending-video",
            original_name="上午巡检.mp4",
            context={
                "_attachment_preview": {
                    "preview_path": "outputs/previews/browser.mp4",
                    "poster_path": "outputs/previews/poster.jpg",
                }
            },
        )
        detected = self.service.chat("检测这段视频", session_id="pending-video")
        history = self.service.history("pending-video")

        self.assertEqual(received["reply"], "我已接收到视频，请给我下一步指令。")
        self.assertTrue(detected["ok"])
        self.assertEqual(detected["data"]["event_count"], 2)
        self.assertEqual(history[0]["content"], "已发送视频")
        self.assertEqual(history[0]["metadata"]["attachment"]["path"], str(self.video))
        self.assertEqual(
            received["attachment"]["preview_path"], "outputs/previews/browser.mp4"
        )

    def test_image_detection_is_routed_and_persisted(self) -> None:
        detected = self.service.chat(
            "检测这张图片",
            session_id="image-session",
            context={"image_path": str(self.image)},
        )
        previous = self.service.chat("查询上一轮结果", session_id="image-session")

        self.assertTrue(detected["ok"])
        self.assertNotIn("报警编号", detected["reply"])
        self.assertEqual(detected["intent"], "detect_image")
        self.assertEqual(detected["data"]["detection_count"], 2)
        self.assertEqual(detected["data"]["candidate_count"], 1)
        self.assertEqual(detected["data"]["alarm_report"], "测试图片报警报告")
        self.assertEqual(
            detected["data"]["event_frames"],
            [{"event_id": 1, "key_frame": "outputs/frames/image_event_1.jpg"}],
        )
        self.assertEqual(previous["data"]["source_type"], "image")
        self.assertEqual(previous["data"]["detection_count"], 2)

    def test_detection_routes_ignore_unrelated_conversation_context(self) -> None:
        image_result = self.service.chat(
            "检测这个图片",
            session_id="context-image",
            context={
                "image_path": str(self.image),
                "detection_id": "old-detection",
                "task_id": "running-realtime-task",
                "_attachment_preview": {"preview_path": "outputs/preview.jpg"},
            },
        )
        video_result = self.service.chat(
            "检测视频",
            session_id="context-video",
            context={
                "video_path": str(self.video),
                "detection_id": "old-detection",
                "task_id": "running-realtime-task",
            },
        )

        self.assertTrue(image_result["ok"])
        self.assertEqual(image_result["intent"], "detect_image")
        self.assertTrue(video_result["ok"])
        self.assertEqual(video_result["intent"], "detect_video")

    def test_image_detection_without_image_requests_attachment(self) -> None:
        response = self.service.chat("检测这张图片", session_id="image-session")
        self.assertFalse(response["ok"])
        self.assertTrue(response["requires_attachment"])
        self.assertEqual(response["reply"], "还没有看到图片/视频，发过来立刻帮你分析。")

    def test_alarm_can_be_confirmed_then_cancelled_and_is_audited(self) -> None:
        self.service.chat(
            "检测这段视频", session_id="s1", context={"video_path": str(self.video)}
        )
        confirmed = self.service.chat("确认报警", session_id="s1")
        cancelled = self.service.chat("取消报警", session_id="s1")

        self.assertEqual(confirmed["data"]["alarm_status"], "confirmed")
        self.assertEqual(cancelled["data"]["alarm_status"], "cancelled")
        self.assertEqual(
            self.control_calls,
            [("confirm", "alarm-test-001"), ("cancel", "alarm-test-001")],
        )
        alarm = self.store.get_alarm("alarm-test-001")
        self.assertIsNotNone(alarm)
        self.assertEqual(alarm.status, "cancelled")

    def test_messages_are_persisted(self) -> None:
        self.service.chat("你能做什么", session_id="s1")
        history = self.service.history("s1")
        self.assertEqual([item["role"] for item in history], ["user", "assistant"])
        self.assertEqual(history[-1]["intent"], "project_knowledge")
        self.assertEqual(history[-1]["metadata"]["mode"], "knowledge")

    def test_agent_output_hides_timezone_suffix_and_uses_local_wall_time(self) -> None:
        formatted = self.service._format_output_times(
            {
                "local": "2026-07-22T16:38:47+08:00",
                "utc": "2026-07-22T08:38:47+00:00",
                "reply": "开始时间：2026-07-22T16:38:47+08:00",
            }
        )

        self.assertEqual(formatted["local"], "2026-07-22 16:38:47")
        self.assertEqual(formatted["utc"], "2026-07-22 16:38:47")
        self.assertEqual(formatted["reply"], "开始时间：2026-07-22 16:38:47")
        self.assertNotIn("+08:00", str(formatted))

    def test_history_also_formats_preexisting_timestamp_messages(self) -> None:
        self.store.record_message(
            "time-history", "assistant", "结束时间：2026-07-22T16:40:46+08:00"
        )

        history = self.service.history("time-history")

        self.assertEqual(history[0]["content"], "结束时间：2026-07-22 16:40:46")


if __name__ == "__main__":
    unittest.main()
