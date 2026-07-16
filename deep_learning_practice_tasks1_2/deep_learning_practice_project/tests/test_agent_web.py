from __future__ import annotations

import io
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from web_app import app


class FakeAgentService:
    def __init__(self) -> None:
        self.chat_calls = []
        self.attachment_calls = []
        self.history_calls = []
        self.history_messages = [{"role": "assistant", "content": "历史消息"}]

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

    def receive_attachment(
        self,
        media_type,
        media_path,
        *,
        session_id="default",
        original_name="",
        context=None,
    ):
        self.attachment_calls.append(
            {
                "media_type": media_type,
                "media_path": media_path,
                "session_id": session_id,
                "original_name": original_name,
                "context": context or {},
            }
        )
        return {
            "ok": True,
            "session_id": session_id,
            "intent": "attachment_received",
            "reply": f"我已接收到{'图片' if media_type == 'image' else '视频'}，请给我下一步指令。",
            "data": {"media_type": media_type},
            "attachment_received": True,
        }

    def history(self, session_id="default", limit=50):
        self.history_calls.append({"session_id": session_id, "limit": limit})
        return self.history_messages


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
        preview_assets = {
            "preview_path": "outputs/agent_inputs/video_previews/saved_browser.mp4",
            "poster_path": "outputs/agent_inputs/video_previews/saved_poster.jpg",
        }
        with (
            patch("web_app.save_uploaded_video", return_value=saved_video),
            patch("web_app.create_browser_video_assets", return_value=preview_assets),
        ):
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
        self.assertEqual(context["_attachment_preview"], preview_assets)

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

    def test_attachment_can_be_sent_without_a_text_instruction(self) -> None:
        saved_image = Path("outputs/agent_inputs/images/saved.jpg")
        with patch("web_app.save_uploaded_image_file", return_value=saved_image):
            response = self.client.post(
                "/api/agent/chat",
                data={
                    "message": "",
                    "session_id": "attachment-session",
                    "media": (io.BytesIO(b"fake-image"), "belt.jpg"),
                },
                content_type="multipart/form-data",
            )

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["attachment_received"])
        self.assertEqual(payload["reply"], "我已接收到图片，请给我下一步指令。")
        self.assertEqual(self.service.chat_calls, [])
        self.assertEqual(self.service.attachment_calls[0]["media_path"], str(saved_image))

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

    def test_history_endpoint_backfills_rule_generated_alarm_report(self) -> None:
        self.service.history_messages = [
            {
                "role": "assistant",
                "content": "检测完成",
                "metadata": {"data": {"detection_id": "det-old"}},
            }
        ]
        record = Mock(
            alarm_report="二、报警结论\n高风险\n\n六、处理建议\n立即停机。",
            source_type="video",
            summary={
                "events": [
                    {"event_id": 1, "key_frame": "outputs/frames/event_1.jpg"}
                ]
            },
        )
        with patch(
            "web_app.agent_history_store.get_detection", return_value=record
        ):
            response = self.client.get(
                "/api/agent/history?session_id=web-session&limit=20"
            )

        data = response.get_json()["messages"][0]["metadata"]["data"]
        self.assertIn("二、报警结论", data["alarm_report"])
        self.assertEqual(data["event_frames"][0]["event_id"], 1)

    def test_invalid_chat_and_history_requests_return_400(self) -> None:
        empty_chat = self.client.post("/api/agent/chat", data={"message": ""})
        invalid_limit = self.client.get("/api/agent/history?limit=0")

        self.assertEqual(empty_chat.status_code, 400)
        self.assertEqual(invalid_limit.status_code, 400)


if __name__ == "__main__":
    unittest.main()
