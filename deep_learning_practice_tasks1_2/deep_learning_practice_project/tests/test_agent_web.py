from __future__ import annotations

import io
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from web_app import app


class FakeAgentService:
    def __init__(self) -> None:
        self.chat_calls = []
        self.history_calls = []

    def chat(self, message, *, session_id="default", context=None):
        self.chat_calls.append(
            {"message": message, "session_id": session_id, "context": context or {}}
        )
        return {
            "ok": True,
            "session_id": session_id,
            "intent": "previous_result",
            "confidence": 1.0,
            "recognizer_source": "hybrid_rules",
            "tool_name": "history_query",
            "reply": "测试回复",
            "data": {},
        }

    def history(self, session_id="default", limit=50):
        self.history_calls.append({"session_id": session_id, "limit": limit})
        return [{"role": "assistant", "content": "历史消息"}]


class AgentWebIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_service = app.config.get("AGENT_SERVICE")
        self.service = FakeAgentService()
        app.config.update(TESTING=True, AGENT_SERVICE=self.service)
        self.client = app.test_client()

    def tearDown(self) -> None:
        app.config["AGENT_SERVICE"] = self.previous_service

    def test_homepage_mounts_agent_component(self) -> None:
        response = self.client.get("/")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn('data-mode="agent"', html)
        self.assertIn('data-agent-chat', html)
        self.assertIn('agent_chat/agent_chat.js', html)
        self.assertIn('name="media"', html)

    def test_chat_endpoint_forwards_message_and_session(self) -> None:
        response = self.client.post(
            "/api/agent/chat",
            data={"message": "查询上一轮结果", "session_id": "web-session"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["ok"])
        self.assertEqual(self.service.chat_calls[0]["session_id"], "web-session")
        self.assertEqual(self.service.chat_calls[0]["message"], "查询上一轮结果")

    def test_chat_endpoint_saves_video_and_builds_context(self) -> None:
        saved_video = Path("outputs/uploaded_videos/saved.mp4")
        with patch("web_app.save_uploaded_video", return_value=saved_video):
            response = self.client.post(
                "/api/agent/chat",
                data={
                    "message": "检测这段视频",
                    "session_id": "video-session",
                    "video_start_time": "2026-07-16T10:00:00",
                    "video": (io.BytesIO(b"fake-video"), "belt.mp4"),
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        context = self.service.chat_calls[0]["context"]
        self.assertEqual(context["video_path"], str(saved_video))
        self.assertEqual(context["video_start_time"], "2026-07-16T10:00:00")

    def test_chat_endpoint_routes_image_attachment(self) -> None:
        saved_image = Path("outputs/web_inputs/saved.jpg")
        with patch("web_app.save_uploaded_image_file", return_value=saved_image):
            response = self.client.post(
                "/api/agent/chat",
                data={
                    "message": "检测这张图片",
                    "session_id": "image-session",
                    "media": (io.BytesIO(b"fake-image"), "belt.jpg"),
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        context = self.service.chat_calls[0]["context"]
        self.assertEqual(context["image_path"], str(saved_image))
        self.assertNotIn("video_path", context)

    def test_image_upload_with_long_name_is_safely_truncated(self) -> None:
        uploaded = Mock()
        uploaded.filename = f"{'very_long_image_name_' * 20}.jpg"

        from web_app import save_uploaded_image_file

        saved_path = save_uploaded_image_file(uploaded)

        self.assertLessEqual(len(saved_path.stem), 80 + 23)
        self.assertEqual(saved_path.suffix, ".jpg")
        uploaded.save.assert_called_once_with(saved_path)

    def test_history_endpoint_returns_persisted_messages(self) -> None:
        response = self.client.get(
            "/api/agent/history?session_id=web-session&limit=20"
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["messages"][0]["content"], "历史消息")
        self.assertEqual(
            self.service.history_calls[0], {"session_id": "web-session", "limit": 20}
        )

    def test_invalid_chat_and_history_requests_return_400(self) -> None:
        empty_chat = self.client.post("/api/agent/chat", data={"message": ""})
        invalid_limit = self.client.get("/api/agent/history?limit=0")

        self.assertEqual(empty_chat.status_code, 400)
        self.assertEqual(invalid_limit.status_code, 400)


if __name__ == "__main__":
    unittest.main()
