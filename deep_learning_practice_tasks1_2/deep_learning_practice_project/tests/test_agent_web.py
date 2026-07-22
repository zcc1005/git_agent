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
        self.skill_calls = []
        self.history_messages = [{"role": "assistant", "content": "历史消息"}]
        self.monitoring_task_id = "monitor-abcdef123456"
        self.realtime_task_id = "realtime-abcdef123456"

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

    def run_skill(self, skill_name, *, session_id="default", arguments=None):
        arguments = dict(arguments or {})
        self.skill_calls.append(
            {
                "skill_name": skill_name,
                "session_id": session_id,
                "arguments": arguments,
            }
        )
        if skill_name == "start-monitoring-task":
            return {
                "skill_name": skill_name,
                "ok": True,
                "reply": "监控任务已创建。",
                "data": {
                    "task_id": self.monitoring_task_id,
                    "source_id": arguments.get("source_id", "main-monitor"),
                    "status": "scheduled",
                    "monitoring_job": {
                        "task_id": self.monitoring_task_id,
                        "status": "pending",
                    },
                },
            }
        if skill_name == "control-stream-archive":
            action = arguments.get("action")
            status = {"start": "starting", "stop": "stopping"}.get(action, "running")
            return {
                "skill_name": skill_name,
                "ok": True,
                "reply": "录像归档操作完成。",
                "data": {
                    "found": True,
                    "source_id": arguments.get("source_id", "main-monitor"),
                    "status": status,
                    "segment_seconds": arguments.get("segment_seconds", 60),
                    "retention_hours": arguments.get("retention_hours", 24),
                    "segments": [
                        {
                            "segment_id": "archive-segment-1",
                            "status": "ready",
                            "started_at": "2026-07-21T00:00:00+00:00",
                            "ended_at": "2026-07-21T00:01:00+00:00",
                        }
                    ],
                },
            }
        if skill_name == "start-realtime-inspection":
            return {"skill_name": skill_name, "ok": True, "reply": "正在启动实时巡检...",
                    "data": {"task_id": self.realtime_task_id, "source_id": "main-monitor", "status": "scheduled"}}
        if skill_name == "control-realtime-inspection":
            if arguments.get("action") == "stop":
                return {"skill_name": skill_name, "ok": True, "reply": "已请求停止实时巡检。",
                        "data": {"task_id": self.realtime_task_id, "status": "stop_requested"}}
            if not arguments.get("task_id"):
                return {"skill_name": skill_name, "ok": True, "reply": "找到实时巡检任务。",
                        "data": {"found": True, "tasks": [{"task_id": self.realtime_task_id}]}}
            return {"skill_name": skill_name, "ok": True, "reply": "实时巡检运行中。",
                    "data": {"found": True, "task": {"task_id": self.realtime_task_id,
                        "source_id": "main-monitor", "status": "running",
                        "start_time": "2026-07-22T08:00:00+08:00", "end_time": "2026-07-22T08:02:00+08:00",
                        "elapsed_seconds": 30, "frames_read": 750, "frames_inferred": 60,
                        "inference_fps": 2.0, "events_detected": 1, "alarms_created": 1,
                        "highest_risk_level": "high", "reconnect_count": 0,
                        "latest_event_frame": "outputs/realtime/event_0001.jpg"},
                        "events": [{"event_id": "event-1", "image_path": "outputs/realtime/event_0001.jpg"}]}}
        if skill_name != "control-monitoring-task":
            raise AssertionError(f"unexpected skill: {skill_name}")
        if arguments.get("action") == "stop":
            return {
                "skill_name": skill_name,
                "ok": True,
                "reply": "已请求停止监控任务。",
                "data": {
                    "task_id": self.monitoring_task_id,
                    "status": "stop_requested",
                    "monitoring_job": {
                        "task_id": self.monitoring_task_id,
                        "status": "stopping",
                    },
                },
            }
        if not arguments.get("task_id"):
            return {
                "skill_name": skill_name,
                "ok": True,
                "reply": "找到监控任务。",
                "data": {
                    "found": True,
                    "tasks": [{"task_id": self.monitoring_task_id}],
                },
            }
        return {
            "skill_name": skill_name,
            "ok": True,
            "reply": "监控任务运行中。",
            "data": {
                "found": True,
                "task": {
                    "task_id": self.monitoring_task_id,
                    "source_id": "main-monitor",
                    "status": "running",
                    "start_time": "2026-07-21T08:00:00+08:00",
                    "end_time": "2026-07-21T09:00:00+08:00",
                    "runs_completed": 1,
                    "runs_succeeded": 1,
                    "runs_failed": 0,
                    "last_alarm_id": "",
                    "last_detection_id": "det-web-monitor",
                    "last_risk_level": "high",
                    "last_error_message": "",
                },
                "monitoring_job": {
                    "task_id": self.monitoring_task_id,
                    "source_id": "main-monitor",
                    "status": "connecting",
                    "started_at": "2026-07-21T08:00:00+08:00",
                    "ends_at": "2026-07-21T09:00:00+08:00",
                    "last_processed_at": "2026-07-21T08:01:00+08:00",
                    "last_error": "",
                    "updated_at": "2026-07-21T08:01:01+08:00",
                },
                "segments": [
                    {
                        "segment_id": "segment-new",
                        "task_id": self.monitoring_task_id,
                        "source_id": "main-monitor",
                        "video_path": "outputs/rtsp/new.mp4",
                        "started_at": "2026-07-21T00:01:00+00:00",
                        "ended_at": "2026-07-21T00:02:00+00:00",
                        "status": "processing",
                        "detection_id": "",
                        "retry_count": 0,
                    },
                    {
                        "segment_id": "segment-old",
                        "task_id": self.monitoring_task_id,
                        "source_id": "main-monitor",
                        "video_path": "outputs/rtsp/old.mp4",
                        "started_at": "2026-07-21T00:00:00+00:00",
                        "ended_at": "2026-07-21T00:01:00+00:00",
                        "status": "completed",
                        "detection_id": "det-web-monitor",
                        "retry_count": 0,
                    },
                ],
                "runs": [{"run_index": 1, "status": "succeeded"}],
            },
        }


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
        self.assertIn('data-agent-monitoring', html)
        self.assertIn('/api/agent/monitoring/events', html)
        self.assertIn('data-realtime-events-endpoint="/api/agent/realtime-inspection/events"', html)
        self.assertNotIn('data-agent-realtime-source', html)
        self.assertNotIn('<strong>持续实时巡检</strong>', html)
        self.assertNotIn('data-agent-followups', html)
        self.assertNotIn('data-agent-followup=', html)
        self.assertNotIn('id="agentPhaseTrack"', html)
        self.assertNotIn('<span class="active">理解</span>', html)

    def test_realtime_frontend_uses_sqlite_cursor_dedupe_and_updates_closed_card(self) -> None:
        script = (
            Path(__file__).resolve().parents[1] / "static" / "agent_chat" / "agent_chat.js"
        ).read_text(encoding="utf-8")

        self.assertIn('after_event_id', script)
        self.assertIn('realtimeEventKey(event)', script)
        self.assertIn('displayedRealtimeEvents.has(key)', script)
        self.assertIn('event.event_status === "closed" ? "已关闭" : "持续中"', script)
        self.assertIn('realtimeReportAnnouncedTaskId !== String(task.task_id)', script)
        self.assertIn('new CustomEvent("agent:realtime-event"', script)

    def test_knowledge_reference_is_rendered_as_compact_final_line(self) -> None:
        script = (
            Path(__file__).resolve().parents[1] / "static" / "agent_chat" / "agent_chat.js"
        ).read_text(encoding="utf-8")
        stylesheet = (
            Path(__file__).resolve().parents[1] / "static" / "agent_chat" / "agent_chat.css"
        ).read_text(encoding="utf-8")

        self.assertIn('referenceMatch = normalizedText.match', script)
        self.assertIn('agent-chat__knowledge-reference', script)
        self.assertIn('.agent-chat__knowledge-reference', stylesheet)

    def test_dashboard_records_realtime_events_without_verbose_task_copy(self) -> None:
        script = (
            Path(__file__).resolve().parents[1] / "static" / "web_app.js"
        ).read_text(encoding="utf-8")

        self.assertIn('function createRealtimeEventRecord(event, task = {})', script)
        self.assertIn('sourceType: "agent"', script)
        self.assertIn('eventCount: 1', script)
        self.assertIn('saveHistoryRecord(createRealtimeEventRecord(item, task))', script)
        self.assertIn('正在实时巡检${source}，时间：${start} 至 ${end}。', script)
        self.assertIn('item?.sourceName === "智能体任务"', script)
        self.assertNotIn('setAgentPhase(', script)

    def test_chat_endpoint_forwards_message_and_session(self) -> None:
        response = self.client.post(
            "/api/agent/chat",
            data={"message": "查询上一轮结果", "session_id": "web-session"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["ok"])
        self.assertEqual(self.service.chat_calls[0]["session_id"], "web-session")
        self.assertEqual(self.service.chat_calls[0]["message"], "查询上一轮结果")

    def test_chat_endpoint_forwards_latest_detection_id_for_followup(self) -> None:
        response = self.client.post(
            "/api/agent/chat",
            data={
                "message": "为什么是高风险？",
                "session_id": "web-session",
                "detection_id": "det-current-123",
                "task_id": "realtime-current-123",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            self.service.chat_calls[0]["context"]["detection_id"],
            "det-current-123",
        )
        self.assertEqual(
            self.service.chat_calls[0]["context"]["task_id"],
            "realtime-current-123",
        )

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

    def test_monitoring_start_endpoint_returns_immediately_with_polling_urls(self) -> None:
        response = self.client.post(
            "/api/agent/monitoring/start",
            json={
                "session_id": "web-session",
                "source_id": "main-monitor",
                "run_duration_seconds": 600,
                "capture_duration_seconds": 30,
                "interval_seconds": 60,
            },
        )

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["data"]["task_id"], self.service.monitoring_task_id)
        self.assertEqual(payload["polling"]["recommended_interval_ms"], 2000)
        call = self.service.skill_calls[0]
        self.assertEqual(call["skill_name"], "start-monitoring-task")
        self.assertEqual(call["session_id"], "web-session")
        self.assertNotIn("session_id", call["arguments"])

    def test_realtime_inspection_start_stop_status_and_events_endpoints(self) -> None:
        started = self.client.post(
            "/api/agent/realtime-inspection/start",
            json={"session_id": "web-session", "source_id": "main-monitor",
                  "run_duration_seconds": 120, "sample_fps": 2},
        )
        self.assertEqual(started.status_code, 200)
        self.assertEqual(started.get_json()["data"]["task_id"], self.service.realtime_task_id)
        status = self.client.get(
            f"/api/agent/realtime-inspection/status?session_id=web-session&task_id={self.service.realtime_task_id}"
        )
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.get_json()["data"]["task"]["frames_inferred"], 60)
        events = self.client.get(
            f"/api/agent/realtime-inspection/events?session_id=web-session&task_id={self.service.realtime_task_id}"
        )
        self.assertEqual(len(events.get_json()["data"]["events"]), 1)
        stopped = self.client.post(
            "/api/agent/realtime-inspection/stop",
            json={"session_id": "web-session", "task_id": self.service.realtime_task_id},
        )
        self.assertEqual(stopped.status_code, 200)
        self.assertEqual(stopped.get_json()["data"]["status"], "stop_requested")

    def test_realtime_event_endpoint_forwards_incremental_filters(self) -> None:
        cursor = f"{self.service.realtime_task_id}-event-0001"
        response = self.client.get(
            "/api/agent/realtime-inspection/events"
            f"?session_id=web-session&task_id={self.service.realtime_task_id}"
            f"&after_event_id={cursor}&active_only=true&limit=25"
        )

        self.assertEqual(response.status_code, 200)
        call = self.service.skill_calls[-1]
        self.assertEqual(call["skill_name"], "control-realtime-inspection")
        self.assertEqual(call["arguments"]["after_event_id"], cursor)
        self.assertTrue(call["arguments"]["active_only"])
        self.assertTrue(call["arguments"]["events_only"])
        self.assertEqual(call["arguments"]["limit"], 25)

    def test_monitoring_stop_endpoint_uses_control_skill(self) -> None:
        response = self.client.post(
            "/api/agent/monitoring/stop",
            json={
                "session_id": "web-session",
                "task_id": self.service.monitoring_task_id,
            },
        )

        self.assertEqual(response.status_code, 200)
        call = self.service.skill_calls[0]
        self.assertEqual(call["skill_name"], "control-monitoring-task")
        self.assertEqual(call["arguments"]["action"], "stop")
        self.assertEqual(call["arguments"]["task_id"], self.service.monitoring_task_id)

    def test_monitoring_status_returns_connection_segment_and_progress(self) -> None:
        response = self.client.get(
            "/api/agent/monitoring/status"
            f"?session_id=web-session&task_id={self.service.monitoring_task_id}"
        )

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["found"])
        self.assertEqual(payload["status"], "connecting")
        self.assertEqual(payload["connection"]["label"], "正在连接/采集")
        self.assertEqual(payload["current_segment"]["segment_id"], "segment-new")
        self.assertEqual(payload["progress"]["estimated_percent"], 50)
        self.assertEqual(payload["progress"]["runs_completed"], 1)

    def test_monitoring_events_support_incremental_segment_cursor(self) -> None:
        response = self.client.get(
            "/api/agent/monitoring/events"
            f"?session_id=web-session&task_id={self.service.monitoring_task_id}"
            "&after_segment_id=segment-old"
        )

        payload = response.get_json()
        segment_events = [
            item for item in payload["events"] if item["event_type"] == "stream_segment"
        ]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(segment_events), 1)
        self.assertEqual(segment_events[0]["segment_id"], "segment-new")
        self.assertEqual(payload["next_cursor"], "segment-new")
        self.assertEqual(payload["current_segment"]["status"], "processing")

    def test_monitoring_endpoint_rejects_unknown_fields_and_invalid_limit(self) -> None:
        invalid_start = self.client.post(
            "/api/agent/monitoring/start",
            json={"source_id": "main-monitor", "rtsp_url": "rtsp://secret/live"},
        )
        invalid_events = self.client.get("/api/agent/monitoring/events?limit=0")

        self.assertEqual(invalid_start.status_code, 400)
        self.assertEqual(invalid_events.status_code, 400)

    def test_archive_start_stop_status_and_segments_use_archive_skill(self) -> None:
        started = self.client.post(
            "/api/agent/archive/start",
            json={
                "session_id": "web-session",
                "source_id": "main-monitor",
                "segment_seconds": 60,
                "retention_hours": 24,
            },
        )
        status = self.client.get(
            "/api/agent/archive/status?session_id=web-session&source_id=main-monitor"
        )
        segments = self.client.get(
            "/api/agent/archive/segments?session_id=web-session&source_id=main-monitor&limit=50"
        )
        stopped = self.client.post(
            "/api/agent/archive/stop",
            json={"session_id": "web-session", "source_id": "main-monitor"},
        )

        self.assertEqual(started.status_code, 200)
        self.assertEqual(status.status_code, 200)
        self.assertEqual(segments.status_code, 200)
        self.assertEqual(stopped.status_code, 200)
        self.assertEqual(segments.get_json()["data"]["segments"][0]["status"], "ready")
        archive_calls = [
            call
            for call in self.service.skill_calls
            if call["skill_name"] == "control-stream-archive"
        ]
        self.assertEqual(
            [call["arguments"]["action"] for call in archive_calls],
            ["start", "query", "query", "stop"],
        )

    def test_archive_api_rejects_rtsp_url_and_missing_source_id(self) -> None:
        invalid_start = self.client.post(
            "/api/agent/archive/start",
            json={"source_id": "main-monitor", "rtsp_url": "rtsp://secret/live"},
        )
        missing_source = self.client.get("/api/agent/archive/status")

        self.assertEqual(invalid_start.status_code, 400)
        self.assertEqual(missing_source.status_code, 400)


if __name__ == "__main__":
    unittest.main()
