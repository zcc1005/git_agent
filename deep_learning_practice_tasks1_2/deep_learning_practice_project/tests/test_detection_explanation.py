from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from agent import AgentService, AgentTools, ImageDetectionOutcome
from storage import SQLiteHistoryStore


FIXED_NOW = datetime(2026, 7, 22, 9, 30, tzinfo=timezone.utc)


class FakeExplainer:
    def __init__(self, *, summary: str = "", fail: bool = False) -> None:
        self.summary = summary or (
            "本次在上传图片中检测到石块异物1个，最高置信度0.9100，规则引擎判定为高风险。"
            "主要原因是检测框面积达到规则升级阈值，建议立即报警并按既定流程停机复核、清理异物后再恢复运行。"
        )
        self.fail = fail
        self.summary_calls = []
        self.explain_calls = []

    def summarize_detection(self, facts):
        self.summary_calls.append(dict(facts))
        if self.fail:
            raise RuntimeError("LLM unavailable")
        return self.summary

    def explain_detection(self, question, question_type, facts, history):
        self.explain_calls.append(
            {
                "question": question,
                "question_type": question_type,
                "facts": dict(facts),
                "history": list(history),
            }
        )
        if self.fail:
            raise RuntimeError("LLM unavailable")
        return f"基于检测记录 {facts['detection_id']} 回答：规则风险仍为{facts['risk_level_name']}。"


class DetectionExplanationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.image = self.root / "belt.jpg"
        self.image.write_bytes(b"image")
        self.store = SQLiteHistoryStore(
            self.root / "history.sqlite3", now=lambda: FIXED_NOW
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def _alarm_document() -> dict:
        return {
            "report_id": "alarm-explanation",
            "source": {
                "type": "image",
                "name": "belt.jpg",
                "start_real_time": "2026-07-22T17:30:00+08:00",
            },
            "detection_summary": {
                "event_count": 1,
                "detection_box_count": 1,
                "class_counts": {"石块异物": 1},
            },
            "events": [
                {
                    "event_id": 1,
                    "key_frame": "outputs/frames/stone.jpg",
                    "objects": [
                        {
                            "class": "stone",
                            "class_name": "石块异物",
                            "confidence": 0.91,
                            "position": "画面中部",
                            "bbox_xyxy": [10, 20, 110, 160],
                        }
                    ],
                    "detection_summary": {
                        "max_confidence": 0.91,
                        "class_counts": {"石块异物": 1},
                    },
                    "risk": {
                        "level": "high",
                        "reason": "石块异物检测框面积达到大目标阈值。",
                    },
                }
            ],
            "overall_risk": {
                "level": "high",
                "reason": "事件1达到高风险，总体风险按最高事件确定。",
                "requires_stop": True,
            },
            "generated_report": {
                "recommended_action": "立即报警并停止皮带，人工清理后复核设备再恢复运行。"
            },
        }

    def _service(self, explainer) -> AgentService:
        def image_runner(image_path, parameters):
            del parameters
            detection = {
                "status": "detected",
                "source": str(image_path),
                "num_detections": 1,
                "num_candidates": 0,
                "class_counts": {"石块异物": 1},
                "objects": [
                    {
                        "class": "stone",
                        "class_name": "石块异物",
                        "confidence": 0.91,
                        "bbox_xyxy": [10, 20, 110, 160],
                    }
                ],
            }
            return ImageDetectionOutcome(
                detection,
                self._alarm_document(),
                "完整规则报告",
                visualization_image="outputs/frames/stone.jpg",
            )

        tools = AgentTools(
            self.store,
            image_detection_runner=image_runner,
            detection_explainer=explainer,
            now=lambda: FIXED_NOW,
        )
        return AgentService(self.store, tools=tools)

    def _detect(self, service: AgentService, session_id: str = "operator") -> dict:
        return service.chat(
            "检测这张图片",
            session_id=session_id,
            context={"image_path": str(self.image)},
        )

    def test_detection_response_contains_structured_alert_and_llm_summary(self) -> None:
        explainer = FakeExplainer()
        result = self._detect(self._service(explainer))

        self.assertTrue(result["ok"])
        presentation = result["data"]
        facts = presentation["structured_alert"]
        self.assertEqual(presentation["analysis_source"], "llm")
        self.assertIn("AI", "AI 智能简析")
        self.assertEqual(facts["risk_level"], "high")
        self.assertEqual(facts["alarm_status"], "pending")
        self.assertEqual(facts["class_counts"], {"石块异物": 1})
        self.assertEqual(facts["max_confidence"], 0.91)
        self.assertEqual(len(presentation["quick_questions"]), 4)
        self.assertEqual(explainer.summary_calls[0]["detection_id"], facts["detection_id"])

    def test_llm_failure_uses_fallback_without_affecting_detection(self) -> None:
        result = self._detect(self._service(FakeExplainer(fail=True)))

        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["analysis_source"], "fallback")
        self.assertTrue(result["data"]["ai_analysis"])
        self.assertEqual(result["data"]["structured_alert"]["risk_level"], "high")

    def test_conflicting_llm_summary_is_rejected_and_facts_remain_authoritative(self) -> None:
        conflicting = (
            "本次检测应改为低风险，并且报警状态已经取消。虽然记录里有石块异物，"
            "但模型重新判断后认为无需采用原有规则结果，建议把正式结构化风险和报警状态一并修改为新的结论。"
        )
        result = self._detect(self._service(FakeExplainer(summary=conflicting)))

        self.assertEqual(result["data"]["analysis_source"], "fallback")
        facts = result["data"]["structured_alert"]
        self.assertEqual(facts["risk_level"], "high")
        self.assertEqual(facts["alarm_status"], "pending")
        self.assertEqual(facts["class_counts"], {"石块异物": 1})

    def test_four_contextual_followups_use_latest_detection_id(self) -> None:
        explainer = FakeExplainer()
        service = self._service(explainer)
        detected = self._detect(service)
        detection_id = detected["data"]["detection_id"]
        questions = {
            "为什么是高风险？": "risk_reason",
            "有什么处置建议？": "action_advice",
            "查看同类历史": "similar_history",
            "解释目标位置": "target_position",
        }

        for question, expected_type in questions.items():
            result = service.chat(
                question,
                session_id="operator",
                context={"detection_id": detection_id},
            )
            self.assertTrue(result["ok"])
            self.assertEqual(result["tool_name"], "explain-detection-result")
            self.assertEqual(result["data"]["detection_id"], detection_id)
            self.assertEqual(result["data"]["question_type"], expected_type)

        self.assertEqual(
            [call["facts"]["detection_id"] for call in explainer.explain_calls],
            [detection_id] * 4,
        )

    def test_followup_without_detection_requests_detection_first(self) -> None:
        service = self._service(FakeExplainer())
        result = service.chat("为什么是高风险？", session_id="empty-session")

        self.assertFalse(result["ok"])
        self.assertTrue(result["data"]["needs_detection"])
        self.assertIn("请先执行一次", result["reply"])


if __name__ == "__main__":
    unittest.main()
