from __future__ import annotations

import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from agent.realtime_inspection import (
    ActiveEvent,
    LatestFrameBuffer,
    RealtimeInspectionError,
    RealtimeInspectionManager,
    StreamingEventAggregator,
)
from agent.service import AgentService
from agent.skills.builtin import create_builtin_skill_registry
from agent.tools import AgentTools
from agent.video_sources import LongVideoSource, LongVideoSourceRegistry, RtspStreamSettings, VideoZone
from storage import SQLiteHistoryStore


class FakeFrame:
    def __init__(self, value: int = 0) -> None:
        self.value = value

    def copy(self):
        return FakeFrame(self.value)


class FakeReader:
    def __init__(self, *, open_results=None, read_results=None, read_delay=0.002) -> None:
        self.open_results = list(open_results or [True])
        self.read_results = list(read_results or [])
        self.read_delay = read_delay
        self.open_calls = 0
        self.read_calls = 0
        self.release_calls = 0

    def open(self):
        self.open_calls += 1
        return self.open_results.pop(0) if self.open_results else True

    def read(self):
        self.read_calls += 1
        if self.read_delay:
            time.sleep(self.read_delay)
        if self.read_results:
            ok = self.read_results.pop(0)
            return ok, FakeFrame(self.read_calls) if ok else None
        return True, FakeFrame(self.read_calls)

    def release(self):
        self.release_calls += 1


def source(kind="rtsp", zones=()):
    return LongVideoSource(
        source_id="main-monitor", display_name="皮带主监控", source_kind=kind,
        video_path="data/test.mp4" if kind == "file" else "", started_at=None,
        line_id="main-line", zones=tuple(zones),
        stream=RtspStreamSettings(url_env="MAIN_RTSP_URL") if kind == "rtsp" else None,
    )


def config(**overrides):
    values = {
        "parameters": {}, "reconnect_interval_seconds": 0.01,
        "max_consecutive_failures": 3, "min_event_hits": 2,
        "event_silence_seconds": 0.03,
    }
    values.update(overrides)
    return values


class RealtimeInspectionCoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = SQLiteHistoryStore(Path(self.temp.name) / "history.sqlite3")

    def tearDown(self):
        self.temp.cleanup()

    def manager(self, reader, detector=None, sink=None):
        detector = detector or (lambda frame, offset: {"objects": []})
        return RealtimeInspectionManager(
            self.store, reader_factory=lambda _: reader,
            detector_factory=lambda _: detector, event_sink=sink,
            recover_orphans=False,
        )

    def test_latest_frame_buffer_is_bounded_and_drops_old_frames(self):
        buffer = LatestFrameBuffer(1)
        now = datetime.now(timezone.utc)
        buffer.put_latest((FakeFrame(1), now, 0.0))
        buffer.put_latest((FakeFrame(2), now, 0.1))
        frame, _, _ = buffer.queue.get_nowait()
        self.assertEqual(buffer.maxsize, 1)
        self.assertEqual(buffer.dropped, 1)
        self.assertEqual(frame.value, 2)

    def test_streaming_aggregator_confirms_immediately_then_closes_same_event(self):
        aggregator = StreamingEventAggregator(min_event_hits=2, silence_seconds=0.1)
        now = datetime.now(timezone.utc)
        obj = {"class": "stone", "class_name": "石块异物", "confidence": 0.8,
               "bbox_xyxy": [10, 10, 40, 40], "track_id": 1}
        self.assertEqual(aggregator.update([obj], FakeFrame(), now), [])
        confirmed = aggregator.update([obj], FakeFrame(), now + timedelta(milliseconds=30))
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].change_type, "confirmed")
        self.assertEqual(confirmed[0].event_status, "active")
        completed = aggregator.expire(now + timedelta(seconds=1))
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].change_type, "closed")
        self.assertEqual(completed[0].event_status, "closed")
        self.assertEqual(completed[0].hit_count, 2)
        self.assertEqual(aggregator.expire(now + timedelta(seconds=2)), [])

    def test_immediate_task_samples_instead_of_inferring_every_frame_and_releases(self):
        reader = FakeReader(read_delay=0.001)
        inferred = []
        manager = self.manager(reader, detector=lambda frame, offset: inferred.append(frame.value) or {"objects": []})
        now = datetime.now(timezone.utc)
        task = manager.start_task(
            "s1", source=source(), start_time=now, end_time=now + timedelta(seconds=0.35),
            zone_id="", sample_fps=10, config=config(),
        )
        terminal = manager.wait_for_terminal(task.id, 2)
        self.assertEqual(terminal.status, "completed")
        self.assertGreater(terminal.frames_read, terminal.frames_inferred)
        self.assertLessEqual(terminal.frames_inferred, 6)
        self.assertGreaterEqual(reader.release_calls, 1)
        self.assertFalse(any(thread.name == f"realtime-inspection-{task.id}" for thread in threading.enumerate()))

    def test_future_task_is_scheduled_then_runs(self):
        reader = FakeReader()
        manager = self.manager(reader)
        now = datetime.now(timezone.utc)
        task = manager.start_task(
            "s1", source=source(), start_time=now + timedelta(seconds=0.08),
            end_time=now + timedelta(seconds=0.22), zone_id="", sample_fps=2, config=config(),
        )
        self.assertEqual(task.status, "scheduled")
        self.assertEqual(manager.wait_for_terminal(task.id, 2).status, "completed")

    def test_slow_inference_drops_frames_and_does_not_grow_queue(self):
        reader = FakeReader(read_delay=0.001)
        def slow_detector(frame, offset):
            time.sleep(0.18)
            return {"objects": []}
        manager = self.manager(reader, slow_detector)
        now = datetime.now(timezone.utc)
        task = manager.start_task("s1", source=source(), start_time=now,
                                  end_time=now + timedelta(seconds=0.45), zone_id="",
                                  sample_fps=10, config=config())
        terminal = manager.wait_for_terminal(task.id, 3)
        self.assertGreater(terminal.frames_dropped, 0)
        self.assertLess(terminal.frames_inferred, terminal.frames_read)

    def test_confirmed_event_creates_once_while_updates_and_close_are_persisted(self):
        reader = FakeReader(read_delay=0.003)
        calls = []
        def detector(frame, offset):
            return {"objects": [{"class": "stone", "class_name": "石块异物", "confidence": 0.81,
                                  "bbox_xyxy": [1, 1, 20, 20], "track_id": 7}]}
        def sink(task, event):
            calls.append((event.change_type, event.hit_count))
            return {"risk_level": "high", "detection_id": "det-1", "alarm_id": "alarm-1",
                    "image_path": "outputs/realtime/event.jpg"}
        manager = self.manager(reader, detector, sink)
        now = datetime.now(timezone.utc)
        task = manager.start_task("s1", source=source(), start_time=now,
                                  end_time=now + timedelta(seconds=0.36), zone_id="",
                                  sample_fps=10, config=config(event_silence_seconds=0.2))
        terminal = manager.wait_for_terminal(task.id, 3)
        self.assertEqual(calls[0][0], "confirmed")
        self.assertEqual(calls[-1][0], "closed")
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(terminal.events_detected, 1)
        self.assertEqual(terminal.alarms_created, 1)
        self.assertEqual(terminal.highest_risk_level, "high")

    def test_query_stop_and_session_isolation(self):
        reader = FakeReader()
        manager = self.manager(reader)
        now = datetime.now(timezone.utc)
        task = manager.start_task("owner", source=source(), start_time=now,
                                  end_time=now + timedelta(seconds=5), zone_id="",
                                  sample_fps=2, config=config())
        with self.assertRaises(RealtimeInspectionError) as caught:
            manager.stop_task(task.id, session_id="other")
        self.assertEqual(caught.exception.code, "task_not_found")
        requested = manager.stop_task(task.id, session_id="owner")
        self.assertEqual(requested.status, "stop_requested")
        self.assertEqual(manager.wait_for_terminal(task.id, 2).status, "stopped")

    def test_disconnect_reconnects_and_recovers(self):
        reader = FakeReader(read_results=[False, True, True], read_delay=0.002)
        manager = self.manager(reader)
        now = datetime.now(timezone.utc)
        task = manager.start_task("s1", source=source(), start_time=now,
                                  end_time=now + timedelta(seconds=0.2), zone_id="",
                                  sample_fps=2, config=config())
        terminal = manager.wait_for_terminal(task.id, 2)
        self.assertEqual(terminal.status, "completed")
        self.assertGreaterEqual(terminal.reconnect_count, 1)
        self.assertGreaterEqual(reader.open_calls, 2)

    def test_consecutive_read_failures_end_as_failed(self):
        reader = FakeReader(read_results=[False] * 10, read_delay=0)
        manager = self.manager(reader)
        now = datetime.now(timezone.utc)
        task = manager.start_task("s1", source=source(), start_time=now,
                                  end_time=now + timedelta(seconds=2), zone_id="",
                                  sample_fps=2, config=config(max_consecutive_failures=2))
        terminal = manager.wait_for_terminal(task.id, 2)
        self.assertEqual(terminal.status, "failed")
        self.assertEqual(terminal.last_error_code, "consecutive_failures_exceeded")

    def test_start_rejects_past_end_over_24_hours_and_gpu_busy(self):
        reader = FakeReader()
        manager = self.manager(reader)
        now = datetime.now(timezone.utc)
        cases = [
            (now - timedelta(seconds=2), now - timedelta(seconds=1)),
            (now, now + timedelta(hours=25)),
        ]
        for start, end in cases:
            with self.subTest(start=start, end=end), self.assertRaises(RealtimeInspectionError):
                manager.start_task("s1", source=source(), start_time=start, end_time=end,
                                   zone_id="", sample_fps=2, config=config())
        active = manager.start_task("s1", source=source(), start_time=now,
                                    end_time=now + timedelta(seconds=5), zone_id="",
                                    sample_fps=2, config=config())
        with self.assertRaises(RealtimeInspectionError) as caught:
            manager.start_task("s2", source=source(), start_time=now,
                               end_time=now + timedelta(seconds=2), zone_id="", sample_fps=2, config=config())
        self.assertEqual(caught.exception.code, "gpu_busy")
        manager.stop_task(active.id, session_id="s1")
        manager.wait_for_terminal(active.id, 2)

    def test_store_marks_orphans_interrupted(self):
        now = datetime.now(timezone.utc)
        task = self.store.create_realtime_inspection_task(
            "s1", source_id="main-monitor", line_id="main-line", zone_id="",
            start_time=now, end_time=now + timedelta(minutes=1), sample_fps=2, config=config(),
        )
        self.store.update_realtime_inspection_task(task.id, status="running")
        self.assertEqual(self.store.interrupt_active_realtime_inspections(), 1)
        recovered = self.store.get_realtime_inspection_task(task.id)
        self.assertEqual(recovered.status, "interrupted")
        self.assertEqual(recovered.last_error_code, "task_interrupted")

    def test_real_event_persistence_saves_representative_frame_json_alarm_and_no_mp4(self):
        import numpy as np

        now = datetime.now(timezone.utc)
        task = self.store.create_realtime_inspection_task(
            "s1", source_id="main-monitor", line_id="main-line", zone_id="",
            start_time=now, end_time=now + timedelta(minutes=1), sample_fps=2, config=config(),
        )
        manager = self.manager(FakeReader())
        tools = AgentTools(self.store, realtime_inspection_manager=manager)
        obj = {"class": "stone", "class_name": "石块异物", "class_id": 0,
               "predicted_class": "stone", "predicted_class_name": "石块异物",
               "confidence": 0.81, "bbox_xyxy": [5, 5, 45, 45], "track_id": 1,
               "detection_state": "confirmed_known", "rejected_as_unknown": False}
        event = ActiveEvent(1, "石块异物", "stone", now, now + timedelta(seconds=1),
                            [5, 5, 45, 45], 0.81, 2,
                            np.zeros((64, 64, 3), dtype=np.uint8), obj, 1)
        output_root = Path(self.temp.name) / "outputs"
        with patch("agent.tools.OUTPUTS_DIR", output_root):
            metadata = tools._persist_realtime_event(task, event)
        image = Path(metadata["image_path"])
        if not image.is_absolute():
            image = Path.cwd() / image
        event_dir = output_root / "realtime_inspections" / "main-monitor" / task.id / "events"
        self.assertTrue((event_dir / "event_0001.jpg").is_file())
        self.assertTrue((event_dir / "event_0001.json").is_file())
        self.assertTrue(metadata["detection_id"])
        self.assertTrue(metadata["alarm_id"])
        self.assertEqual(len(self.store.list_realtime_inspection_events(task.id)), 1)
        self.assertEqual(list(output_root.rglob("*.mp4")), [])

    def test_event_is_queryable_before_task_finishes_and_reuses_detection_alarm(self):
        import numpy as np

        reader = FakeReader(read_delay=0.002)
        confirmed = threading.Event()
        obj = {"class": "stone", "class_name": "石块异物", "class_id": 0,
               "predicted_class": "stone", "predicted_class_name": "石块异物",
               "confidence": 0.84, "bbox_xyxy": [5, 5, 45, 45], "track_id": 9,
               "detection_state": "confirmed_known", "rejected_as_unknown": False}
        detector = lambda frame, offset: {"objects": [obj]}
        manager = self.manager(reader, detector)
        tools = AgentTools(self.store, realtime_inspection_manager=manager)

        def immediate_sink(task, event):
            result = tools._persist_realtime_event(task, event)
            if event.change_type == "confirmed":
                confirmed.set()
            return result

        manager._event_sink = immediate_sink
        now = datetime.now(timezone.utc)
        output_root = Path(self.temp.name) / "outputs-live"
        with (
            patch("agent.tools.OUTPUTS_DIR", output_root),
            patch(
                "tests.test_realtime_inspection.FakeReader.read",
                lambda instance: (
                    time.sleep(instance.read_delay) or True,
                    np.zeros((64, 64, 3), dtype=np.uint8),
                ),
            ),
        ):
            task = manager.start_task(
                "s1", source=source(), start_time=now,
                end_time=now + timedelta(seconds=1.5), zone_id="",
                sample_fps=10, config=config(event_silence_seconds=0.4),
            )
            self.assertTrue(confirmed.wait(1.0))
            running = self.store.get_realtime_inspection_task(task.id)
            events = self.store.list_realtime_inspection_events(task.id)
            self.assertIn(running.status, {"running", "reconnecting"})
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].event_status, "active")
            self.assertIsNotNone(self.store.get_detection(events[0].detection_id))
            self.assertIsNotNone(self.store.get_alarm(events[0].alarm_id))
            first_detection_id, first_alarm_id = events[0].detection_id, events[0].alarm_id
            manager.stop_task(task.id, session_id="s1")
            terminal = manager.wait_for_terminal(task.id, 2)

        closed = self.store.list_realtime_inspection_events(task.id)
        self.assertEqual(terminal.status, "stopped")
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].event_status, "closed")
        self.assertEqual(closed[0].detection_id, first_detection_id)
        self.assertEqual(closed[0].alarm_id, first_alarm_id)
        self.assertEqual(terminal.events_detected, 1)
        self.assertEqual(terminal.alarms_created, 1)

    def test_active_event_updates_best_confidence_and_reappearing_object_is_new(self):
        aggregator = StreamingEventAggregator(min_event_hits=2, silence_seconds=0.1)
        now = datetime.now(timezone.utc)
        first = {"class": "stone", "class_name": "石块异物", "confidence": 0.7,
                 "bbox_xyxy": [10, 10, 40, 40], "track_id": 1}
        best = {**first, "confidence": 0.93}
        aggregator.update([first], FakeFrame(1), now)
        confirmed = aggregator.update([first], FakeFrame(2), now + timedelta(milliseconds=20))[0]
        updated = aggregator.update([best], FakeFrame(9), now + timedelta(milliseconds=40))[0]
        self.assertEqual(confirmed.sequence, updated.sequence)
        self.assertEqual(updated.change_type, "updated")
        self.assertEqual(updated.confidence, 0.93)
        self.assertEqual(updated.representative_frame.value, 9)
        aggregator.expire(now + timedelta(seconds=1))
        aggregator.update([first], FakeFrame(3), now + timedelta(seconds=1.1))
        second = aggregator.update([first], FakeFrame(4), now + timedelta(seconds=1.12))[0]
        self.assertNotEqual(second.sequence, confirmed.sequence)

    def test_after_event_id_returns_only_newer_events(self):
        now = datetime.now(timezone.utc)
        task = self.store.create_realtime_inspection_task(
            "s1", source_id="main-monitor", line_id="main-line", zone_id="",
            start_time=now, end_time=now + timedelta(minutes=1), sample_fps=2,
            config=config(),
        )
        ids = []
        for sequence in (1, 2, 3):
            event_id = f"{task.id}-event-{sequence:04d}"
            ids.append(event_id)
            self.store.record_realtime_inspection_event(
                event_id=event_id, task_id=task.id, source_id=task.source_id,
                detected_at=(now + timedelta(seconds=sequence)).isoformat(),
                ended_at=(now + timedelta(seconds=sequence)).isoformat(),
                class_name="石块异物", confidence=0.8, bbox=[1, 1, 10, 10],
                risk_level="high", detection_id=f"det-{sequence}", alarm_id=f"alarm-{sequence}",
                image_path=f"event-{sequence}.jpg", metadata={}, event_status="active",
            )
        newer = self.store.list_realtime_inspection_events(
            task.id, after_event_id=ids[0], limit=10
        )
        self.assertEqual([item.event_id for item in newer], ids[1:])

    def test_terminal_summaries_distinguish_completed_stopped_and_failed(self):
        base = {
            "task_id": "realtime-abcdef123456", "source_id": "main-monitor",
            "start_time": "2026-07-22T08:00:00+08:00",
            "end_time": "2026-07-22T08:01:00+08:00", "elapsed_seconds": 60,
            "frames_read": 100, "frames_inferred": 20, "events_detected": 1,
            "alarms_created": 1, "highest_risk_level": "high",
        }
        completed = AgentTools._realtime_terminal_summary({**base, "status": "completed"})
        stopped = AgentTools._realtime_terminal_summary({**base, "status": "stopped"})
        failed = AgentTools._realtime_terminal_summary(
            {**base, "status": "failed", "last_error_code": "connection_failed",
             "last_error_message": "连接失败"}
        )
        self.assertIn("【实时巡检已结束】", completed)
        self.assertIn("达到计划结束时间", completed)
        self.assertIn("【实时巡检已停止】", stopped)
        self.assertIn("用户主动停止", stopped)
        self.assertIn("【实时巡检异常结束】", failed)
        self.assertIn("安全错误码：connection_failed", failed)

    def test_realtime_llm_failure_keeps_durable_rule_summary(self):
        attempted = threading.Event()

        class FailingExplainer:
            def summarize_detection(self, facts):
                attempted.set()
                raise RuntimeError("llm offline")

        now = datetime.now(timezone.utc)
        task = self.store.create_realtime_inspection_task(
            "s1", source_id="main-monitor", line_id="main-line", zone_id="",
            start_time=now, end_time=now + timedelta(minutes=1), sample_fps=2,
            config=config(),
        )
        event_id = f"{task.id}-event-0001"
        self.store.record_realtime_inspection_event(
            event_id=event_id, task_id=task.id, source_id=task.source_id,
            detected_at=now.isoformat(), ended_at=now.isoformat(),
            class_name="石块异物", confidence=0.9, bbox=[1, 1, 10, 10],
            risk_level="high", detection_id="det-1", alarm_id="alarm-1",
            image_path="event.jpg", metadata={"analysis_source": "fallback"},
            event_status="active", llm_summary="规则模板简析",
        )
        tools = AgentTools(
            self.store, realtime_inspection_manager=self.manager(FakeReader()),
            detection_explainer=FailingExplainer(),
        )
        tools._schedule_realtime_event_summary(
            event_id,
            {"class_counts": {"石块异物": 1}, "risk_level": "high",
             "risk_level_name": "高风险", "alarm_status": "pending"},
        )
        self.assertTrue(attempted.wait(0.5))
        record = self.store.get_realtime_inspection_event(event_id)
        self.assertEqual(record.llm_summary, "规则模板简析")
        self.assertEqual(record.metadata["analysis_source"], "fallback")


class RealtimeInspectionSkillTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = SQLiteHistoryStore(Path(self.temp.name) / "history.sqlite3")
        self.registry_data = LongVideoSourceRegistry(sources=(source(zones=(VideoZone("zone-a", "A区", (0, 0, 100, 100)),)),))
        # Do not let AgentTools create actual background workers in validation-only tests.
        self.reader = FakeReader()
        self.manager = RealtimeInspectionManager(
            self.store, reader_factory=lambda _: self.reader,
            detector_factory=lambda _: (lambda frame, offset: {"objects": []}),
            recover_orphans=False,
        )
        self.tools = AgentTools(
            self.store, video_source_registry_loader=lambda: self.registry_data,
            realtime_inspection_manager=self.manager,
        )
        self.skills = create_builtin_skill_registry(self.tools)

    def tearDown(self):
        for task in self.store.list_realtime_inspection_tasks(session_id="s1", statuses=("scheduled", "connecting", "running", "reconnecting", "stop_requested"), limit=10):
            try:
                self.manager.stop_task(task.id, session_id="s1")
                self.manager.wait_for_terminal(task.id, 2)
            except RealtimeInspectionError:
                pass
        self.temp.cleanup()

    def test_schema_requires_exactly_one_end_condition(self):
        for arguments in (
            {"source_id": "main-monitor"},
            {"source_id": "main-monitor", "end_time": "2026-07-23T10:00:00+08:00", "run_duration_seconds": 60},
        ):
            with self.subTest(arguments=arguments):
                result = self.skills.invoke("start-realtime-inspection", session_id="s1", arguments=arguments)
                self.assertFalse(result.ok)
                self.assertEqual(result.error_code, "invalid_arguments")

    def test_schema_rejects_zone_roi_conflict_and_forbidden_fields(self):
        result = self.skills.invoke("start-realtime-inspection", session_id="s1", arguments={
                "source_id": "main-monitor", "run_duration_seconds": 60,
                "zone_id": "zone-a", "parameters": {"roi": [0, 0, 10, 10]},
            })
        self.assertFalse(result.ok)
        for field in ("rtsp_url", "line_id", "output_path", "model_path", "command"):
            with self.subTest(field=field):
                result = self.skills.invoke("start-realtime-inspection", session_id="s1", arguments={
                    "source_id": "main-monitor", "run_duration_seconds": 60, field: "forbidden",
                })
                self.assertFalse(result.ok)

    def test_non_rtsp_source_and_missing_zone_return_safe_errors(self):
        file_registry = LongVideoSourceRegistry(sources=(source(kind="file"),))
        tools = AgentTools(self.store, video_source_registry_loader=lambda: file_registry,
                           realtime_inspection_manager=self.manager)
        result = tools.start_realtime_inspection("s1", {"source_id": "main-monitor", "run_duration_seconds": 10})
        self.assertEqual(result["error_code"], "not_rtsp_source")
        result = self.tools.start_realtime_inspection("s1", {
            "source_id": "main-monitor", "run_duration_seconds": 10, "zone_id": "missing",
        })
        self.assertEqual(result["error_code"], "zone_not_found")
        missing = self.tools.start_realtime_inspection("s1", {
            "source_id": "missing", "run_duration_seconds": 10,
        })
        self.assertEqual(missing["error_code"], "source_not_found")

    def test_control_aliases_normalize_to_query(self):
        for alias in ("view", "show", "status", "get"):
            result = self.skills.invoke("control-realtime-inspection", session_id="s1", arguments={"action": alias})
            self.assertTrue(result.ok)
        result = self.skills.invoke("control-realtime-inspection", session_id="s1", arguments={"action": "delete"})
        self.assertFalse(result.ok)

    def test_query_without_task_id_returns_latest_task_details(self):
        now = datetime.now(timezone.utc)
        task = self.store.create_realtime_inspection_task(
            "s1", source_id="main-monitor", line_id="main-line", zone_id="",
            start_time=now, end_time=now + timedelta(minutes=2), sample_fps=2,
            config=config(),
        )
        self.store.update_realtime_inspection_task(
            task.id, status="running", started_at=now.isoformat(),
            frames_read=250, frames_inferred=20, events_detected=1,
            alarms_created=1, highest_risk_level="high",
        )

        result = self.skills.invoke(
            "control-realtime-inspection", session_id="s1", arguments={"action": "query"}
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.data["task"]["task_id"], task.id)
        self.assertEqual(result.data["task"]["frames_read"], 250)
        self.assertEqual(result.data["task"]["display_name"], "皮带主监控")
        self.assertIn("读取250帧", result.reply)
        self.assertIn("推理20帧", result.reply)

    def test_task_only_query_does_not_build_event_reports(self):
        now = datetime.now(timezone.utc)
        task = self.store.create_realtime_inspection_task(
            "s1", source_id="main-monitor", line_id="main-line", zone_id="",
            start_time=now, end_time=now + timedelta(minutes=2), sample_fps=2,
            config=config(),
        )
        self.store.update_realtime_inspection_task(
            task.id, status="running", frames_read=250, frames_inferred=20,
            events_detected=1, alarms_created=1, highest_risk_level="high",
        )
        self.store.record_realtime_inspection_event(
            event_id=f"{task.id}-event-0001", task_id=task.id,
            source_id="main-monitor", detected_at=now.isoformat(),
            ended_at="", class_name="石块异物", confidence=0.91,
            bbox=[10, 20, 100, 120], risk_level="high",
            detection_id="det-realtime-1", alarm_id="alarm-realtime-1",
            image_path="outputs/realtime/event_0001.jpg", metadata={},
        )

        with patch.object(
            self.store,
            "list_realtime_inspection_events",
            side_effect=AssertionError("状态轮询不应读取事件详情"),
        ):
            result = self.skills.invoke(
                "control-realtime-inspection",
                session_id="s1",
                arguments={
                    "action": "query",
                    "task_id": task.id,
                    "task_only": True,
                },
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.data["task"]["frames_read"], 250)
        self.assertEqual(result.data["events"], [])
        self.assertNotIn("realtime_report", result.data)

    def test_compact_event_query_omits_internal_alarm_document(self):
        now = datetime.now(timezone.utc)
        task = self.store.create_realtime_inspection_task(
            "s1", source_id="main-monitor", line_id="main-line", zone_id="",
            start_time=now, end_time=now + timedelta(minutes=2), sample_fps=2,
            config=config(),
        )
        detection, alarm = self.store.record_detection(
            "s1", source_type="realtime",
            source_path=f"realtime://main-monitor/{task.id}",
            detection={"status": "completed", "class_counts": {"石块异物": 1}},
            alarm_document={
                "report_id": "alarm-compact",
                "overall_risk": {"level": "high", "requires_stop": True},
            },
            alarm_report="测试报警报告",
        )
        self.store.record_realtime_inspection_event(
            event_id=f"{task.id}-event-0001", task_id=task.id,
            source_id="main-monitor", detected_at=now.isoformat(),
            ended_at="", class_name="石块异物", confidence=0.91,
            bbox=[10, 20, 100, 120], risk_level="high",
            detection_id=detection.id, alarm_id=alarm.id,
            image_path="outputs/realtime/event_0001.jpg",
            metadata={"alarm_json_path": "outputs/realtime/alarm.json"},
        )

        result = self.skills.invoke(
            "control-realtime-inspection",
            session_id="s1",
            arguments={
                "action": "query",
                "task_id": task.id,
                "events_only": True,
                "compact": True,
            },
        )

        self.assertTrue(result.ok)
        event = result.data["events"][0]
        self.assertEqual(event["alarm_status"], "pending")
        self.assertEqual(event["alarm_report"]["text"], "测试报警报告")
        self.assertNotIn("document", event["alarm_report"])
        self.assertNotIn("metadata", event)
        self.assertNotIn("realtime_report", result.data)

    def test_natural_language_status_query_bypasses_slow_llm_planner(self):
        now = datetime.now(timezone.utc)
        task = self.store.create_realtime_inspection_task(
            "s1", source_id="main-monitor", line_id="main-line", zone_id="",
            start_time=now, end_time=now + timedelta(minutes=2), sample_fps=2,
            config=config(),
        )

        class PlannerMustNotRun:
            def plan(self, message, *, catalog, context):
                raise AssertionError("明确的实时巡检状态查询不应调用大模型规划器")

        service = AgentService(
            self.store, tools=self.tools, skill_planner=PlannerMustNotRun()
        )
        result = service.chat("查询当前实时巡检任务状态", session_id="s1")

        self.assertTrue(result["ok"])
        self.assertEqual(result["recognizer_source"], "deterministic_realtime_control")
        self.assertEqual(result["data"]["task"]["task_id"], task.id)

    def test_natural_language_stop_accepts_common_typo_and_bypasses_planner(self):
        now = datetime.now(timezone.utc)
        task = self.manager.start_task(
            "s1", source=source(), start_time=now,
            end_time=now + timedelta(seconds=5), zone_id="",
            sample_fps=2, config=config(),
        )

        class PlannerMustNotRun:
            def plan(self, message, *, catalog, context):
                raise AssertionError("明确的停止指令不应调用大模型规划器")

        service = AgentService(
            self.store, tools=self.tools, skill_planner=PlannerMustNotRun()
        )
        result = service.chat("停止实施巡检", session_id="s1")

        self.assertTrue(result["ok"])
        self.assertEqual(result["recognizer_source"], "deterministic_realtime_control")
        self.assertEqual(result["recognition_metadata"]["action"], "stop")
        self.assertEqual(result["data"]["task_id"], task.id)

    def test_stop_is_idempotent_when_latest_task_already_completed(self):
        now = datetime.now(timezone.utc)
        task = self.store.create_realtime_inspection_task(
            "s1", source_id="main-monitor", line_id="main-line", zone_id="",
            start_time=now - timedelta(minutes=2), end_time=now,
            sample_fps=2, config=config(),
        )
        self.store.update_realtime_inspection_task(task.id, status="completed")

        result = self.skills.invoke(
            "control-realtime-inspection", session_id="s1", arguments={"action": "stop"}
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.data["task"]["task_id"], task.id)
        self.assertIn("无需重复停止", result.reply)

    def test_completed_task_query_returns_realtime_alarm_report(self):
        now = datetime.now(timezone.utc)
        task = self.store.create_realtime_inspection_task(
            "s1", source_id="main-monitor", line_id="main-line", zone_id="",
            start_time=now - timedelta(minutes=2), end_time=now,
            sample_fps=2, config=config(),
        )
        self.store.update_realtime_inspection_task(
            task.id, status="completed", events_detected=1, alarms_created=1,
            highest_risk_level="high", frames_read=100, frames_inferred=20,
        )
        self.store.record_realtime_inspection_event(
            event_id=f"{task.id}-event-0001", task_id=task.id,
            source_id="main-monitor", detected_at=now.isoformat(),
            ended_at=now.isoformat(), class_name="石块异物", confidence=0.91,
            bbox=[10, 20, 100, 120], risk_level="high",
            detection_id="det-realtime-1", alarm_id="alarm-realtime-1",
            image_path="outputs/realtime/event_0001.jpg", metadata={},
        )

        result = self.skills.invoke(
            "control-realtime-inspection", session_id="s1", arguments={"action": "query"}
        )

        self.assertTrue(result.ok)
        report = result.data["realtime_report"]
        self.assertEqual(report["task_id"], task.id)
        self.assertEqual(report["event_count"], 1)
        self.assertEqual(report["class_counts"], {"石块异物": 1})
        self.assertEqual(report["risk_level"], "high")
        self.assertEqual(report["events"][0]["detection_id"], "det-realtime-1")
        self.assertTrue(report["ai_analysis"])

    def test_natural_language_realtime_report_bypasses_planner(self):
        now = datetime.now(timezone.utc)
        task = self.store.create_realtime_inspection_task(
            "s1", source_id="main-monitor", line_id="main-line", zone_id="",
            start_time=now - timedelta(minutes=2), end_time=now,
            sample_fps=2, config=config(),
        )
        self.store.update_realtime_inspection_task(task.id, status="completed")

        class PlannerMustNotRun:
            def plan(self, message, *, catalog, context):
                raise AssertionError("明确的实时巡检报告指令不应调用大模型规划器")

        service = AgentService(
            self.store, tools=self.tools, skill_planner=PlannerMustNotRun()
        )
        result = service.chat("输出上一轮实施巡检的报警报告", session_id="s1")

        self.assertTrue(result["ok"])
        self.assertEqual(result["recognizer_source"], "deterministic_realtime_control")
        self.assertEqual(result["data"]["realtime_report"]["task_id"], task.id)

    def test_natural_language_queries_latest_all_active_and_no_event(self):
        now = datetime.now(timezone.utc)
        task = self.store.create_realtime_inspection_task(
            "s1", source_id="main-monitor", line_id="main-line", zone_id="",
            start_time=now, end_time=now + timedelta(minutes=2), sample_fps=2,
            config=config(),
        )
        self.store.update_realtime_inspection_task(task.id, status="running")
        service = AgentService(self.store, tools=self.tools)

        empty = service.chat("查看主监控最近一次异物报告", session_id="s1")
        self.assertEqual(empty["reply"], "当前实时巡检尚未确认异物事件。")

        event_id = f"{task.id}-event-0001"
        self.store.record_realtime_inspection_event(
            event_id=event_id, task_id=task.id, source_id=task.source_id,
            detected_at=now.isoformat(), ended_at=now.isoformat(),
            last_seen_at=now.isoformat(), event_status="active",
            class_name="石块异物", confidence=0.9, max_confidence=0.9,
            bbox=[10, 10, 50, 50], risk_level="high",
            detection_id="det-live-1", alarm_id="alarm-live-1",
            image_path="outputs/realtime/event.jpg", metadata={},
            class_counts={"石块异物": 1}, hit_count=3,
        )
        latest = service.chat("查看主监控最近一次异物报告", session_id="s1")
        all_events = service.chat("查看本次实时巡检发现的所有异物", session_id="s1")
        active = service.chat("查看当前仍在持续的报警", session_id="s1")
        detail = service.chat(f"查看事件 {event_id} 的详细报告", session_id="s1")

        for result in (latest, all_events, active, detail):
            self.assertTrue(result["ok"])
            self.assertEqual(result["recognizer_source"], "deterministic_realtime_event_query")
            self.assertEqual(result["data"]["events"][0]["event_id"], event_id)
        self.assertTrue(active["data"]["events"][0]["event_status"] == "active")

    def test_realtime_report_batch_alarm_confirmation_and_cancellation(self):
        now = datetime.now(timezone.utc)
        task = self.store.create_realtime_inspection_task(
            "s1", source_id="main-monitor", line_id="main-line", zone_id="",
            start_time=now - timedelta(minutes=2), end_time=now,
            sample_fps=2, config=config(),
        )
        self.store.update_realtime_inspection_task(
            task.id, status="completed", events_detected=2, alarms_created=2,
            highest_risk_level="high",
        )
        alarm_ids = []
        for index in (1, 2):
            detection, alarm = self.store.record_detection(
                "s1", source_type="realtime",
                source_path=f"realtime://main-monitor/{task.id}",
                detection={"status": "completed", "class_counts": {"石块异物": 1}},
                alarm_document={
                    "report_id": f"alarm-batch-{index}",
                    "overall_risk": {
                        "level": "high", "requires_stop": True,
                    },
                },
                alarm_report="测试报警报告",
            )
            alarm_ids.append(alarm.id)
            self.store.record_realtime_inspection_event(
                event_id=f"{task.id}-event-{index:04d}", task_id=task.id,
                source_id="main-monitor", detected_at=now.isoformat(),
                ended_at=now.isoformat(), class_name="石块异物", confidence=0.9,
                bbox=[10, 20, 100, 120], risk_level="high",
                detection_id=detection.id, alarm_id=alarm.id,
                image_path=f"outputs/realtime/event_{index:04d}.jpg", metadata={},
            )

        class PlannerMustNotRun:
            def plan(self, message, *, catalog, context):
                raise AssertionError("明确的本轮报警闭环不应调用大模型规划器")

        service = AgentService(
            self.store, tools=self.tools, skill_planner=PlannerMustNotRun()
        )
        before = self.skills.invoke(
            "control-realtime-inspection", session_id="s1", arguments={"action": "query"}
        )
        confirmed = service.chat(
            "确认本轮报警", session_id="s1", context={"task_id": task.id}
        )
        repeated = service.chat(
            "确认本轮报警", session_id="s1", context={"task_id": task.id}
        )
        cancelled = service.chat(
            "取消本轮报警", session_id="s1", context={"task_id": task.id}
        )

        self.assertEqual(
            before.data["realtime_report"]["alarm_status_counts"]["pending"], 2
        )
        self.assertTrue(confirmed["ok"])
        self.assertEqual(
            confirmed["recognizer_source"], "deterministic_realtime_alarm_control"
        )
        self.assertEqual(confirmed["data"]["affected_count"], 2)
        self.assertEqual(repeated["data"]["affected_count"], 0)
        self.assertEqual(repeated["data"]["unchanged_count"], 2)
        self.assertEqual(cancelled["data"]["affected_count"], 2)
        self.assertTrue(all(
            self.store.get_alarm(alarm_id).status == "cancelled"
            for alarm_id in alarm_ids
        ))

    def test_realtime_batch_alarm_control_rejects_other_session(self):
        now = datetime.now(timezone.utc)
        task = self.store.create_realtime_inspection_task(
            "s1", source_id="main-monitor", line_id="main-line", zone_id="",
            start_time=now - timedelta(minutes=2), end_time=now,
            sample_fps=2, config=config(),
        )
        self.store.update_realtime_inspection_task(task.id, status="completed")

        result = self.skills.invoke(
            "control-alarm", session_id="other",
            arguments={
                "action": "confirm", "scope": "realtime_task", "task_id": task.id,
            },
        )

        self.assertFalse(result.ok)
        self.assertFalse(result.data["found"])


if __name__ == "__main__":
    unittest.main()
