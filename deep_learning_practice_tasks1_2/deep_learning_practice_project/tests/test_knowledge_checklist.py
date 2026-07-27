from __future__ import annotations

import re
import unittest

from agent import AgentService, RuleBasedIntentRecognizer
from agent.knowledge_base import ProjectKnowledgeBase
from project_config import PROJECT_ROOT


class KnowledgeChecklistTests(unittest.TestCase):
    def test_manual_checklist_contains_thirty_numbered_questions(self) -> None:
        checklist = (
            PROJECT_ROOT / "docs" / "KNOWLEDGE_QA_TEST_QUESTIONS.md"
        ).read_text(encoding="utf-8")
        rows = re.findall(
            r"^\|\s*(\d+)\s*\|\s*([^|]+)\|\s*([^|]+)\|",
            checklist,
            flags=re.M,
        )

        self.assertEqual([int(number) for number, _, _ in rows], list(range(1, 31)))

    def test_each_manual_question_has_repository_evidence(self) -> None:
        checklist = (
            PROJECT_ROOT / "docs" / "KNOWLEDGE_QA_TEST_QUESTIONS.md"
        ).read_text(encoding="utf-8")
        rows = re.findall(
            r"^\|\s*(\d+)\s*\|\s*([^|]+)\|\s*([^|]+)\|",
            checklist,
            flags=re.M,
        )
        knowledge_base = ProjectKnowledgeBase(PROJECT_ROOT)
        missing: list[str] = []
        for number, _, question in rows:
            hits = knowledge_base.search(question.strip(), limit=5)
            if not hits:
                missing.append(f"{number}. {question.strip()}")
                continue
            self.assertNotEqual(
                hits[0].source.lower(),
                "docs/knowledge_qa_test_questions.md",
            )

        self.assertEqual(missing, [])

    def test_core_implementation_sources_are_indexed(self) -> None:
        sources = set(ProjectKnowledgeBase(PROJECT_ROOT).trusted_sources())

        for required in (
            "agent/knowledge_base.py",
            "agent/llm_api.py",
            "agent/service.py",
            "agent/tools.py",
            "storage/sqlite_store.py",
            "static/agent_chat/agent_chat.js",
            "web_app.py",
        ):
            self.assertIn(required, sources)

    def test_manual_questions_route_to_read_only_knowledge_mode(self) -> None:
        checklist = (
            PROJECT_ROOT / "docs" / "KNOWLEDGE_QA_TEST_QUESTIONS.md"
        ).read_text(encoding="utf-8")
        questions = [
            question.strip()
            for _, _, question in re.findall(
                r"^\|\s*(\d+)\s*\|\s*([^|]+)\|\s*([^|]+)\|",
                checklist,
                flags=re.M,
            )
        ]
        recognizer = RuleBasedIntentRecognizer()
        misrouted: list[str] = []
        for question in questions:
            match = recognizer.recognize(question)
            if not AgentService._should_use_knowledge(
                question,
                match,
                has_knowledge_context=False,
            ):
                misrouted.append(question)

        self.assertEqual(misrouted, [])


if __name__ == "__main__":
    unittest.main()
