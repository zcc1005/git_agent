from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from agent import AgentService, AgentTools, ProjectKnowledgeBase, SkillPlan, SkillPlanStep
from agent.llm_api import (
    LLMAPIConfig,
    OpenAICompatibleClient,
    OpenAICompatibleKnowledgeAnswerer,
)
from storage import SQLiteHistoryStore


FIXED_NOW = datetime(2026, 7, 22, 4, 0, tzinfo=timezone.utc)


class RecordingPlanner:
    def __init__(self, plan: SkillPlan | None = None) -> None:
        self.result = plan or SkillPlan(
            steps=(SkillPlanStep("probe-video-source", {"source_id": "main-monitor"}),)
        )
        self.calls: list[dict] = []

    def plan(self, message, *, catalog, context):
        self.calls.append({"message": message, "catalog": catalog, "context": context})
        return self.result


class RejectingPlanner:
    def plan(self, message, *, catalog, context):
        raise AssertionError("项目知识问题不应进入 Skill Planner")


class StaticKnowledgeAnswerer:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[dict] = []

    def answer_project_question(self, question, evidence, history):
        self.calls.append(
            {"question": question, "evidence": evidence, "history": history}
        )
        return self.answer


class FailingKnowledgeAnswerer:
    def answer_project_question(self, question, evidence, history):
        raise RuntimeError("llm offline")


class ClassifyingKnowledgeAnswerer(StaticKnowledgeAnswerer):
    def __init__(self, answer: str, mode: str, clarification: str = "") -> None:
        super().__init__(answer)
        self.mode = mode
        self.clarification = clarification
        self.classification_calls: list[dict] = []

    def classify_request(self, message, history):
        self.classification_calls.append({"message": message, "history": history})
        return {"mode": self.mode, "clarification": self.clarification}


class ProjectKnowledgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        (root / "README.md").write_text(
            """# 项目说明

## 持续实时巡检

持续实时巡检通过 start-realtime-inspection 保持单个 RTSP 连接，按 sample_fps 抽帧检测。
默认不缓存完整视频，只保存确认事件的代表帧。完整录像应使用 control-stream-archive。

## RTSP 接入

监控源必须先登记在 config/video_sources.json，再使用 probe-video-source 查询在线状态。

## 报警与历史记录

报警记录和检测历史保存在 SQLite 数据库中，分别记录报警状态和 detection_id。
""",
            encoding="utf-8",
        )
        (root / "agent").mkdir()
        (root / "agent" / "realtime_inspection.py").write_text(
            "# 持续实时巡检保持一个 RTSP 连接。\n"
            "class RealtimeInspectionManager:\n    pass\n",
            encoding="utf-8",
        )
        self.knowledge_base = ProjectKnowledgeBase(root)
        self.store = SQLiteHistoryStore(
            root / "history.sqlite3", now=lambda: FIXED_NOW
        )
        self.tools = AgentTools(self.store, now=lambda: FIXED_NOW)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_service(self, **kwargs) -> AgentService:
        return AgentService(
            self.store,
            tools=self.tools,
            knowledge_base=self.knowledge_base,
            **kwargs,
        )

    def test_project_question_uses_repository_evidence_not_skill_planner(self) -> None:
        answerer = StaticKnowledgeAnswerer(
            "持续实时巡检保持一个 RTSP 连接，并按 sample_fps 抽帧。"
        )
        service = self.make_service(
            skill_planner=RejectingPlanner(), knowledge_answerer=answerer
        )

        result = service.chat("持续实时巡检是什么？", session_id="knowledge-1")

        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "knowledge")
        self.assertEqual(result["tool_name"], "project_knowledge")
        self.assertEqual(result["data"]["answer_source"], "llm")
        self.assertEqual(result["data"]["knowledge_sources"][0]["source"], "README.md")
        self.assertIn("参考：项目使用说明", result["reply"])
        self.assertEqual(result["data"]["answer_mode"], "user")
        self.assertNotIn("sample_fps", result["reply"])
        self.assertIn("操作步骤", result["reply"])
        self.assertTrue(answerer.calls[0]["evidence"])

    def test_colloquial_alarm_history_question_enters_knowledge_mode(self) -> None:
        answerer = StaticKnowledgeAnswerer("报警记录保存报警状态，检测历史保存检测编号。")
        service = self.make_service(
            skill_planner=RejectingPlanner(), knowledge_answerer=answerer
        )

        result = service.chat("报警、历史记录都是啥", session_id="knowledge-colloquial")

        self.assertEqual(result["mode"], "knowledge")
        self.assertNotIn("大模型任务规划失败", result["reply"])
        self.assertIn("参考：项目使用说明", result["reply"])

    def test_llm_classifier_routes_unmatched_project_question_to_knowledge(self) -> None:
        answerer = ClassifyingKnowledgeAnswerer(
            "SQLite用于保存报警记录和检测历史。", "knowledge"
        )
        service = self.make_service(
            skill_planner=RejectingPlanner(), knowledge_answerer=answerer
        )

        result = service.chat("SQLite这块都管啥？", session_id="knowledge-classifier")

        self.assertEqual(result["mode"], "knowledge")
        self.assertEqual(len(answerer.classification_calls), 1)
        self.assertEqual(result["data"]["answer_source"], "llm")

    def test_llm_classifier_can_request_clarification_without_calling_skill(self) -> None:
        answerer = ClassifyingKnowledgeAnswerer(
            "不会被使用", "clarify", "请说明你想查询状态还是执行检测。"
        )
        service = self.make_service(
            skill_planner=RejectingPlanner(), knowledge_answerer=answerer
        )

        result = service.chat("帮我弄一下", session_id="knowledge-clarify")

        self.assertTrue(result["needs_clarification"])
        self.assertEqual(result["recognizer_source"], "llm_request_classifier")
        self.assertEqual(result["reply"], "请说明你想查询状态还是执行检测。")

    def test_llm_execute_classification_still_uses_skill_planner(self) -> None:
        answerer = ClassifyingKnowledgeAnswerer("不会被使用", "execute")
        planner = RecordingPlanner()
        service = self.make_service(
            skill_planner=planner, knowledge_answerer=answerer
        )

        with patch.object(
            service,
            "run_skill",
            return_value={"ok": True, "reply": "已查询。", "data": {}},
        ) as run_skill:
            result = service.chat("帮我探一下主监控", session_id="execute-classifier")

        self.assertEqual(result["intent"], "skill_plan")
        self.assertEqual(len(planner.calls), 1)
        run_skill.assert_called_once()

    def test_empty_skill_plan_falls_back_to_repository_knowledge(self) -> None:
        planner = RecordingPlanner(SkillPlan())
        answerer = StaticKnowledgeAnswerer("SQLite保存报警记录和检测历史。")
        service = self.make_service(
            skill_planner=planner, knowledge_answerer=answerer
        )

        result = service.chat(
            "SQLite记录放在什么位置呢", session_id="empty-plan-fallback"
        )

        self.assertEqual(result["mode"], "knowledge")
        self.assertEqual(
            result["recognizer_source"],
            "knowledge_fallback_after_empty_skill_plan",
        )
        self.assertNotIn("没有返回可执行", result["reply"])

    def test_context_followup_resolves_pronoun_to_previous_topic(self) -> None:
        answerer = StaticKnowledgeAnswerer("它默认不缓存完整视频，只保留事件代表帧。")
        service = self.make_service(knowledge_answerer=answerer)

        first = service.chat("持续实时巡检是什么？", session_id="knowledge-context")
        second = service.chat("它会保存视频吗？", session_id="knowledge-context")

        self.assertEqual(first["mode"], "knowledge")
        self.assertEqual(second["mode"], "knowledge")
        self.assertIn("持续实时巡检是什么", second["knowledge_query"])
        self.assertIn("它会保存视频吗", second["knowledge_query"])

    def test_dynamic_online_question_still_calls_skill_planner(self) -> None:
        planner = RecordingPlanner()
        service = self.make_service(skill_planner=planner)
        routed_calls = []

        def fake_run_skill(skill_name, *, session_id="default", arguments=None):
            routed_calls.append((skill_name, session_id, dict(arguments or {})))
            return {"ok": True, "reply": "监控源在线。", "data": {"online": True}}

        with patch.object(service, "run_skill", side_effect=fake_run_skill):
            result = service.chat("主监控在线吗", session_id="dynamic-1")

        self.assertNotEqual(result.get("mode"), "knowledge")
        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(routed_calls[0][0], "probe-video-source")

    def test_how_to_stop_is_knowledge_and_does_not_execute_stop(self) -> None:
        service = self.make_service(skill_planner=RejectingPlanner())
        with patch.object(service, "run_skill") as run_skill:
            result = service.chat("如何查询或停止实时巡检？", session_id="knowledge-stop")

        self.assertEqual(result["mode"], "knowledge")
        run_skill.assert_not_called()

    def test_missing_repository_evidence_requests_clarification(self) -> None:
        empty_root = Path(self.temp_dir.name) / "empty"
        empty_root.mkdir()
        (empty_root / "README.md").write_text("# 项目\n仅有项目标题。", encoding="utf-8")
        service = AgentService(
            self.store,
            tools=self.tools,
            knowledge_base=ProjectKnowledgeBase(empty_root),
            knowledge_answerer=StaticKnowledgeAnswerer("虚构回答"),
        )

        result = service.chat("RTSP量子校准参数在哪里？", session_id="knowledge-missing")

        self.assertEqual(result["mode"], "knowledge")
        self.assertTrue(result["needs_clarification"])
        self.assertEqual(result["data"]["knowledge_sources"], [])
        self.assertIn("现有资料没有说明", result["reply"])

    def test_llm_failure_falls_back_to_repository_excerpt(self) -> None:
        service = self.make_service(knowledge_answerer=FailingKnowledgeAnswerer())

        result = service.chat("sample_fps是什么意思？", session_id="knowledge-fallback")

        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["answer_source"], "fallback")
        self.assertNotIn("根据项目资料", result["reply"])
        self.assertIn("抽帧频率", result["reply"])
        self.assertNotIn("sample_fps", result["reply"])
        self.assertIn("操作步骤", result["reply"])

    def test_llm_invented_path_is_rejected_and_falls_back(self) -> None:
        service = self.make_service(
            knowledge_answerer=StaticKnowledgeAnswerer(
                "配置位于 config/invented_runtime.json。"
            )
        )

        result = service.chat("RTSP如何接入？", session_id="knowledge-safe")

        self.assertEqual(result["data"]["answer_source"], "fallback")
        self.assertNotIn("invented_runtime.json", result["reply"])

    def test_user_answer_hides_internal_names_tables_and_paths(self) -> None:
        answerer = StaticKnowledgeAnswerer(
            "调用 query_history() 查询 detection_runs，说明在 README.md，"
            "返回 pending 和 detection_id。"
        )
        service = self.make_service(knowledge_answerer=answerer)

        result = service.chat("如何查询历史报警？", session_id="knowledge-plain")

        self.assertEqual(result["data"]["answer_mode"], "user")
        for forbidden in (
            "query_history", "detection_runs", "README.md", "pending", "detection_id"
        ):
            self.assertNotIn(forbidden, result["reply"])
        self.assertIn("待处理", result["reply"])
        self.assertIn("检测记录编号", result["reply"])
        self.assertIn("查看今天的报警记录", result["reply"])

    def test_developer_question_can_show_code_location(self) -> None:
        answerer = StaticKnowledgeAnswerer(
            "持续实时巡检主要实现在 agent/realtime_inspection.py。"
        )
        service = self.make_service(knowledge_answerer=answerer)

        result = service.chat(
            "持续实时巡检的代码在哪个文件实现？",
            session_id="knowledge-developer",
        )

        self.assertEqual(result["data"]["answer_mode"], "developer")
        self.assertIn("agent/realtime_inspection.py", result["reply"])
        self.assertIn("参考：", result["reply"])
        self.assertIn("README.md", result["reply"])

    def test_dynamic_skill_reply_is_localized_without_changing_data(self) -> None:
        planner = RecordingPlanner()
        service = self.make_service(skill_planner=planner)
        skill_data = {
            "source_id": "main-monitor",
            "status": "pending",
            "detection_id": "det-123",
        }

        with patch.object(
            service,
            "run_skill",
            return_value={
                "ok": True,
                "reply": "source_id=main-monitor, status=pending, detection_id=det-123",
                "data": skill_data,
            },
        ):
            result = service.chat("查看主监控当前状态", session_id="dynamic-localized")

        self.assertIn("监控源=main-monitor", result["reply"])
        self.assertIn("待处理", result["reply"])
        self.assertIn("检测记录编号=det-123", result["reply"])
        self.assertNotIn("pending", result["reply"])
        self.assertEqual(result["data"]["steps"][0]["data"], skill_data)

    def test_llm_failure_fallback_remains_plain_chinese(self) -> None:
        service = self.make_service(knowledge_answerer=FailingKnowledgeAnswerer())

        result = service.chat("如何查询历史报警？", session_id="fallback-plain")

        self.assertIn("可以直接在聊天框查询历史报警", result["reply"])
        self.assertIn("操作步骤", result["reply"])
        self.assertIn("参考：项目使用说明", result["reply"])
        self.assertNotIn("SQLite", result["reply"])
        self.assertNotIn("detection_id", result["reply"])

    def test_user_answer_sections_are_normalized_to_fixed_order(self) -> None:
        answerer = StaticKnowledgeAnswerer(
            "直接回答：可以查询历史报警。\n注意事项：以实际记录为准。\n操作步骤：输入查询口令。"
        )
        service = self.make_service(knowledge_answerer=answerer)

        result = service.chat("如何查询历史报警？", session_id="ordered-answer")

        reply = result["reply"]
        self.assertLess(reply.index("可以查询历史报警"), reply.index("操作步骤"))
        self.assertLess(reply.index("操作步骤"), reply.index("注意事项"))


class OpenAICompatibleKnowledgeAnswererTests(unittest.TestCase):
    def test_classifier_returns_closed_request_mode(self) -> None:
        calls = []

        def transport(url, body, headers, timeout):
            payload = json.loads(body.decode("utf-8"))
            calls.append(payload)
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"mode": "knowledge", "clarification": ""}
                            )
                        }
                    }
                ]
            }

        answerer = OpenAICompatibleKnowledgeAnswerer(
            OpenAICompatibleClient(
                LLMAPIConfig(
                    api_key="test-key",
                    base_url="https://example.test/v1",
                    model="test-model",
                ),
                transport=transport,
            )
        )

        result = answerer.classify_request("报警记录放在哪里？", [])

        self.assertEqual(result["mode"], "knowledge")
        system_prompt = calls[0]["messages"][0]["content"]
        self.assertIn("execute、dynamic、knowledge、clarify", system_prompt)

    def test_request_is_bounded_to_repository_evidence(self) -> None:
        calls = []

        def transport(url, body, headers, timeout):
            payload = json.loads(body.decode("utf-8"))
            calls.append(payload)
            return {
                "choices": [
                    {"message": {"content": json.dumps({"answer": "按资料回答。"})}}
                ]
            }

        client = OpenAICompatibleClient(
            LLMAPIConfig(
                api_key="test-key",
                base_url="https://example.test/v1",
                model="test-model",
            ),
            transport=transport,
        )
        answerer = OpenAICompatibleKnowledgeAnswerer(client)
        evidence = [
            {
                "source": "README.md",
                "heading": "持续实时巡检",
                "excerpt": "默认不缓存完整视频。",
                "score": 1.0,
            }
        ]

        answer = answerer.answer_project_question(
            "它保存视频吗？",
            evidence,
            [{"role": "user", "content": "持续实时巡检是什么？"}],
        )

        self.assertEqual(answer, "按资料回答。")
        request = json.loads(calls[0]["messages"][1]["content"])
        self.assertEqual(request["repository_evidence"], evidence)
        self.assertEqual(request["answer_mode"], "user")
        self.assertNotIn("video_sources", request)


if __name__ == "__main__":
    unittest.main()
