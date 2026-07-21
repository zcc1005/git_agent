from __future__ import annotations

import tempfile
import threading
import unittest
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent.monitoring import MonitoringTaskManager
from storage import SQLiteHistoryStore


BASE_TIME = datetime(2026, 7, 21, 8, 0, tzinfo=timezone.utc)


class MutableClock:
    def __init__(self, value: datetime) -> None:
        self.value = value
        self.lock = threading.Lock()

    def now(self) -> datetime:
        with self.lock:
            return self.value

    def advance(self, seconds: float) -> None:
        with self.lock:
            self.value += timedelta(seconds=seconds)


class MonitoringTaskManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.clock = MutableClock(BASE_TIME)
        self.store = SQLiteHistoryStore(
            Path(self.temp_dir.name) / "monitoring.sqlite3",
            now=self.clock.now,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def config(**overrides):
        values = {
            "capture_duration_seconds": 1.0,
            "interval_seconds": 0.0,
            "max_consecutive_failures": 3,
            "parameters": {"sample_fps": 1.0},
        }
        values.update(overrides)
        return values

    def test_successful_round_is_persisted_and_task_completes(self) -> None:
        def runner(session_id, arguments):
            self.assertEqual(session_id, "operator")
            self.assertEqual(arguments["source_id"], "main-monitor")
            self.clock.advance(10)
            return {
                "ok": True,
                "reply": "检测完成",
                "data": {
                    "detection_id": "det-round-1",
                    "alarm_id": "alarm-round-1",
                    "risk_level": "high",
                    "event_count": 1,
                    "class_counts": {"石块异物": 1},
                    "capture": {"video_path": "outputs/streams/round-1.mp4"},
                },
            }

        manager = MonitoringTaskManager(
            self.store,
            runner,
            now=self.clock.now,
            recover_orphans=False,
        )
        task = manager.start_task(
            "operator",
            source_id="main-monitor",
            line_id="main-line",
            zone_id="",
            start_time=BASE_TIME,
            end_time=BASE_TIME + timedelta(seconds=5),
            config=self.config(),
        )

        terminal = manager.wait_for_terminal(task.id, timeout_seconds=2)
        runs = self.store.list_monitoring_runs(task.id)
        job = self.store.get_monitoring_job(task.id)
        segments = self.store.list_stream_segments(task.id)

        self.assertEqual(terminal.status, "completed")
        self.assertEqual(terminal.runs_completed, 1)
        self.assertEqual(terminal.runs_succeeded, 1)
        self.assertEqual(terminal.last_detection_id, "det-round-1")
        self.assertEqual(runs[0].status, "succeeded")
        self.assertEqual(runs[0].risk_level, "high")
        self.assertEqual(job.status, "completed")
        self.assertEqual(job.last_processed_at, runs[0].ended_at)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].status, "completed")
        self.assertEqual(segments[0].detection_id, "det-round-1")
        self.assertEqual(segments[0].video_path, "outputs/streams/round-1.mp4")

    def test_stop_request_waits_for_current_round_then_stops(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def runner(session_id, arguments):
            del session_id, arguments
            entered.set()
            release.wait(2)
            return {"ok": True, "reply": "完成", "data": {}}

        manager = MonitoringTaskManager(
            self.store,
            runner,
            now=self.clock.now,
            recover_orphans=False,
        )
        task = manager.start_task(
            "operator",
            source_id="main-monitor",
            line_id="main-line",
            zone_id="",
            start_time=BASE_TIME,
            end_time=BASE_TIME + timedelta(hours=1),
            config=self.config(interval_seconds=60.0),
        )
        self.assertTrue(entered.wait(1))

        requested = manager.stop_task(task.id, session_id="operator")
        self.assertEqual(requested.status, "stop_requested")
        release.set()
        terminal = manager.wait_for_terminal(task.id, timeout_seconds=2)

        self.assertEqual(terminal.status, "stopped")
        self.assertEqual(terminal.runs_completed, 1)
        self.assertEqual(self.store.get_monitoring_job(task.id).status, "cancelled")

    def test_consecutive_failures_stop_at_configured_limit(self) -> None:
        def runner(session_id, arguments):
            del session_id, arguments
            return {
                "ok": False,
                "error_code": "stream_offline",
                "reply": "视频源离线",
                "data": {},
            }

        manager = MonitoringTaskManager(
            self.store,
            runner,
            now=self.clock.now,
            recover_orphans=False,
        )
        task = manager.start_task(
            "operator",
            source_id="main-monitor",
            line_id="main-line",
            zone_id="",
            start_time=BASE_TIME,
            end_time=BASE_TIME + timedelta(hours=1),
            config=self.config(max_consecutive_failures=2),
        )

        terminal = manager.wait_for_terminal(task.id, timeout_seconds=2)

        self.assertEqual(terminal.status, "failed")
        self.assertEqual(terminal.runs_completed, 2)
        self.assertEqual(terminal.runs_failed, 2)
        self.assertEqual(terminal.last_error_code, "stream_offline")
        job = self.store.get_monitoring_job(task.id)
        segments = self.store.list_stream_segments(task.id)
        self.assertEqual(job.status, "failed")
        self.assertIn("连续检测失败", job.last_error)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].status, "failed")
        self.assertEqual(segments[0].retry_count, 1)

    def test_restart_interrupts_active_tasks_without_resuming(self) -> None:
        task = self.store.create_monitoring_task(
            "operator",
            source_id="main-monitor",
            line_id="main-line",
            zone_id="",
            start_time=BASE_TIME.isoformat(),
            end_time=(BASE_TIME + timedelta(hours=1)).isoformat(),
            config=self.config(),
        )
        segment, claimed = self.store.claim_stream_segment(
            task.id,
            source_id="main-monitor",
            started_at=BASE_TIME,
            ended_at=BASE_TIME + timedelta(seconds=1),
        )
        self.assertTrue(claimed)

        MonitoringTaskManager(
            self.store,
            lambda session_id, arguments: {"ok": True, "data": {}},
            now=self.clock.now,
            recover_orphans=True,
        )
        recovered = self.store.get_monitoring_task(task.id)

        self.assertEqual(recovered.status, "interrupted")
        self.assertEqual(recovered.last_error_code, "process_restarted")
        self.assertEqual(self.store.get_monitoring_job(task.id).status, "failed")
        self.assertEqual(
            self.store.list_stream_segments(task.id)[0].status,
            "failed",
        )

    def test_stream_segment_unique_window_is_idempotent_and_retryable(self) -> None:
        task = self.store.create_monitoring_task(
            "operator",
            source_id="main-monitor",
            line_id="main-line",
            zone_id="",
            start_time=BASE_TIME.isoformat(),
            end_time=(BASE_TIME + timedelta(minutes=1)).isoformat(),
            config=self.config(),
        )
        values = {
            "source_id": "main-monitor",
            "started_at": BASE_TIME,
            "ended_at": BASE_TIME + timedelta(seconds=1),
        }

        first, first_claimed = self.store.claim_stream_segment(task.id, **values)
        duplicate, duplicate_claimed = self.store.claim_stream_segment(
            task.id,
            **values,
        )
        self.store.finish_stream_segment(
            first.segment_id,
            succeeded=False,
            video_path="outputs/streams/retry.mp4",
        )
        retry, retry_claimed = self.store.claim_stream_segment(task.id, **values)
        self.store.finish_stream_segment(
            retry.segment_id,
            succeeded=True,
            detection_id="det-retried",
        )
        completed, completed_claimed = self.store.claim_stream_segment(
            task.id,
            **values,
        )

        self.assertTrue(first_claimed)
        self.assertFalse(duplicate_claimed)
        self.assertEqual(first.segment_id, duplicate.segment_id)
        self.assertTrue(retry_claimed)
        self.assertEqual(retry.retry_count, 1)
        self.assertFalse(completed_claimed)
        self.assertEqual(completed.status, "completed")
        self.assertEqual(len(self.store.list_stream_segments(task.id)), 1)

        second, second_claimed = self.store.claim_stream_segment(
            task.id,
            source_id="main-monitor",
            started_at=BASE_TIME + timedelta(seconds=2),
            ended_at=BASE_TIME + timedelta(seconds=3),
        )
        self.assertTrue(second_claimed)
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.finish_stream_segment(
                second.segment_id,
                succeeded=True,
                video_path="outputs/streams/retry.mp4",
                detection_id="det-duplicate-path",
            )

    def test_runtime_tables_have_the_stage_seven_columns(self) -> None:
        with self.store._connect() as connection:
            job_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(monitoring_jobs)"
                ).fetchall()
            }
            segment_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(stream_segments)"
                ).fetchall()
            }

        self.assertEqual(
            job_columns,
            {
                "task_id",
                "source_id",
                "status",
                "started_at",
                "ends_at",
                "segment_seconds",
                "last_processed_at",
                "last_error",
                "created_at",
                "updated_at",
            },
        )
        self.assertEqual(
            segment_columns,
            {
                "segment_id",
                "task_id",
                "source_id",
                "video_path",
                "started_at",
                "ended_at",
                "status",
                "detection_id",
                "retry_count",
                "created_at",
            },
        )

    def test_missing_runtime_job_is_backfilled_for_existing_task(self) -> None:
        task = self.store.create_monitoring_task(
            "operator",
            source_id="main-monitor",
            line_id="main-line",
            zone_id="",
            start_time=BASE_TIME.isoformat(),
            end_time=(BASE_TIME + timedelta(minutes=1)).isoformat(),
            config=self.config(capture_duration_seconds=7.0),
        )
        with self.store._connect() as connection:
            connection.execute(
                "DELETE FROM monitoring_jobs WHERE task_id = ?",
                (task.id,),
            )

        migrated = SQLiteHistoryStore(self.store.db_path, now=self.clock.now)
        job = migrated.get_monitoring_job(task.id)

        self.assertIsNotNone(job)
        self.assertEqual(job.status, "pending")
        self.assertEqual(job.segment_seconds, 7.0)

    def test_stop_request_does_not_move_completed_task_backwards(self) -> None:
        task = self.store.create_monitoring_task(
            "operator",
            source_id="main-monitor",
            line_id="main-line",
            zone_id="",
            start_time=BASE_TIME.isoformat(),
            end_time=(BASE_TIME + timedelta(minutes=1)).isoformat(),
            config=self.config(),
        )
        self.store.update_monitoring_task(task.id, status="completed")

        result = self.store.request_monitoring_task_stop(task.id)

        self.assertEqual(result.status, "completed")

    def test_session_cannot_stop_another_sessions_task(self) -> None:
        manager = MonitoringTaskManager(
            self.store,
            lambda session_id, arguments: {"ok": True, "data": {}},
            now=self.clock.now,
            recover_orphans=False,
        )
        task = manager.start_task(
            "operator-a",
            source_id="main-monitor",
            line_id="main-line",
            zone_id="",
            start_time=BASE_TIME + timedelta(hours=1),
            end_time=BASE_TIME + timedelta(hours=2),
            config=self.config(),
        )

        with self.assertRaises(LookupError):
            manager.stop_task(task.id, session_id="operator-b")
        manager.stop_task(task.id, session_id="operator-a")
        terminal = manager.wait_for_terminal(task.id, timeout_seconds=2)
        self.assertEqual(terminal.status, "stopped")


if __name__ == "__main__":
    unittest.main()
