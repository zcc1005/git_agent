from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from agent import AgentService, AgentTools, ImageDetectionOutcome
from agent.llm_api import (
    LLMAPIConfig,
    OpenAICompatibleClient,
    OpenAICompatibleSkillPlanner,
    create_llm_enabled_service,
    load_env_file,
)
from agent.planners import SkillPlan, SkillPlanStep
from storage import SQLiteHistoryStore


FIXED_NOW = datetime(2026, 7, 16, 10, 30, tzinfo=timezone.utc)


class StaticSkillPlanner:
    def __init__(self, plan: SkillPlan) -> None:
        self.result = plan
        self.calls = []

    def plan(self, message, *, catalog, context):
        self.calls.append(
            {"message": message, "catalog": catalog, "context": context}
        )
        return self.result


class LLMAPIConfigTests(unittest.TestCase):
    def test_config_reads_key_without_hardcoding_it(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LLM_API_KEY": "secret-test-key",
                "LLM_BASE_URL": "https://example.test/v1/",
                "LLM_MODEL": "test-model",
            },
            clear=True,
        ):
            config = LLMAPIConfig.from_env()

        self.assertEqual(config.api_key, "secret-test-key")
        self.assertEqual(config.chat_completions_url, "https://example.test/v1/chat/completions")
        self.assertEqual(config.model, "test-model")

    def test_missing_key_or_model_is_rejected(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "LLM_API_KEY"):
                LLMAPIConfig.from_env()
        with patch.dict(os.environ, {"LLM_API_KEY": "key"}, clear=True):
            with self.assertRaisesRegex(ValueError, "LLM_MODEL"):
                LLMAPIConfig.from_env()

    def test_dotenv_does_not_override_process_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "LLM_API_KEY=file-key\nLLM_MODEL=file-model\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"LLM_API_KEY": "process-key"}, clear=True):
                self.assertTrue(load_env_file(env_file))
                self.assertEqual(os.environ["LLM_API_KEY"], "process-key")
                self.assertEqual(os.environ["LLM_MODEL"], "file-model")


class OpenAICompatiblePlannerTests(unittest.TestCase):
    def test_client_sends_bearer_key_and_parses_json_plan(self) -> None:
        calls = []

        def transport(url, body, headers, timeout):
            calls.append(
                {
                    "url": url,
                    "payload": json.loads(body.decode("utf-8")),
                    "headers": dict(headers),
                    "timeout": timeout,
                }
            )
            content = {
                "summary": "查询一号线历史",
                "needs_clarification": False,
                "clarification": "",
                "steps": [
                    {
                        "skill_name": "query-history",
                        "arguments": {"line_id": "line-1"},
                    }
                ],
            }
            return {"choices": [{"message": {"content": json.dumps(content)}}]}

        config = LLMAPIConfig(
            api_key="secret-key",
            base_url="https://example.test/v1",
            model="planner-model",
        )
        planner = OpenAICompatibleSkillPlanner(
            OpenAICompatibleClient(config, transport=transport)
        )
        plan = planner.plan(
            "查询一号线历史",
            catalog=[{"name": "query-history", "optional_inputs": ["line_id"]}],
            context={"session_id": "s1", "history": [], "request_context": {}},
        )

        self.assertEqual(plan.steps[0].skill_name, "query-history")
        self.assertEqual(plan.steps[0].arguments["line_id"], "line-1")
        self.assertEqual(calls[0]["headers"]["Authorization"], "Bearer secret-key")
        self.assertEqual(calls[0]["payload"]["temperature"], 0)
        self.assertEqual(
            calls[0]["payload"]["response_format"], {"type": "json_object"}
        )

    def test_env_factory_links_model_planner_to_agent_service(self) -> None:
        def transport(url, body, headers, timeout):
            del url, body, headers, timeout
            plan = {
                "summary": "查询线路历史",
                "needs_clarification": False,
                "steps": [
                    {
                        "skill_name": "query-history",
                        "arguments": {"line_id": "line-1", "limit": 10},
                    }
                ],
            }
            return {"choices": [{"message": {"content": json.dumps(plan)}}]}

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_file = root / ".env"
            env_file.write_text(
                "LLM_API_KEY=test-key\n"
                "LLM_BASE_URL=https://example.test/v1\n"
                "LLM_MODEL=test-model\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                service = create_llm_enabled_service(
                    root / "history.sqlite3",
                    env_file=env_file,
                    transport=transport,
                )
                result = service.chat("给我做一个一号线路态势概览")

        self.assertTrue(result["ok"])
        self.assertEqual(result["intent"], "skill_plan")
        self.assertEqual(result["data"]["steps"][0]["skill_name"], "query-history")


class AgentSkillOrchestrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.store = SQLiteHistoryStore(
            self.root / "history.sqlite3", now=lambda: FIXED_NOW
        )
        self.image = self.root / "belt.jpg"
        self.image.write_bytes(b"image")

        def image_runner(image_path, parameters):
            del parameters
            detection = {
                "status": "detected",
                "source": str(image_path),
                "num_images": 1,
                "num_detections": 1,
                "num_candidates": 0,
                "has_foreign_object": True,
                "class_counts": {"石块异物": 1},
                "objects": [{"class": "stone"}],
            }
            alarm = {
                "report_id": "alarm-orchestration",
                "overall_risk": {"level": "medium", "requires_stop": False},
            }
            return ImageDetectionOutcome(detection, alarm, "报警报告")

        self.tools = AgentTools(
            self.store,
            image_detection_runner=image_runner,
            now=lambda: FIXED_NOW,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_unknown_natural_language_is_planned_and_executed(self) -> None:
        planner = StaticSkillPlanner(
            SkillPlan(
                steps=(
                    SkillPlanStep("query-history", {"line_id": "line-7"}),
                ),
                summary="查询线路历史",
            )
        )
        service = AgentService(self.store, tools=self.tools, skill_planner=planner)

        result = service.chat("帮我看看七号线最近的运行态势")

        self.assertTrue(result["ok"])
        self.assertEqual(result["intent"], "skill_plan")
        self.assertEqual(result["data"]["completed_steps"], 1)
        self.assertEqual(planner.calls[0]["context"]["session_id"], "default")

    def test_always_mode_plans_an_existing_simple_intent(self) -> None:
        planner = StaticSkillPlanner(
            SkillPlan(steps=(SkillPlanStep("query-history", {"limit": 5}),))
        )
        service = AgentService(
            self.store,
            tools=self.tools,
            skill_planner=planner,
            skill_planner_mode="always",
        )

        result = service.chat("查询上一轮结果")

        self.assertEqual(result["intent"], "skill_plan")
        self.assertEqual(len(planner.calls), 1)

    def test_multi_step_plan_resolves_previous_detection_id(self) -> None:
        planner = StaticSkillPlanner(
            SkillPlan(
                steps=(
                    SkillPlanStep("detect-image", {}),
                    SkillPlanStep(
                        "review-detection",
                        {
                            "detection_id": "$steps.0.data.detection_id",
                            "action": "close",
                            "note": "已清除",
                        },
                    ),
                )
            )
        )
        service = AgentService(self.store, tools=self.tools, skill_planner=planner)

        result = service.chat(
            "检测这张图片，然后异物已清除并完成闭环",
            session_id="operator",
            context={"image_path": str(self.image)},
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["completed_steps"], 2)
        detection_id = result["data"]["steps"][0]["data"]["detection_id"]
        self.assertEqual(self.store.get_detection(detection_id).review_status, "closed")

    def test_model_cannot_cancel_alarm_from_ambiguous_request(self) -> None:
        detected = AgentService(self.store, tools=self.tools).run_skill(
            "detect-image",
            session_id="operator",
            arguments={"image_path": str(self.image)},
        )
        planner = StaticSkillPlanner(
            SkillPlan(
                steps=(SkillPlanStep("control-alarm", {"action": "cancel"}),)
            )
        )
        service = AgentService(self.store, tools=self.tools, skill_planner=planner)

        result = service.chat("把它处理掉", session_id="operator")

        self.assertFalse(result["ok"])
        self.assertIn("明确动作指令", result["reply"])
        alarm = self.store.get_alarm(detected["data"]["alarm_id"])
        self.assertEqual(alarm.status, "pending")


if __name__ == "__main__":
    unittest.main()
