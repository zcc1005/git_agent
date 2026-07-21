from __future__ import annotations

import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent import AgentService, AgentTools, LongVideoSourceRegistry, VideoDetectionOutcome
from agent.archive import HistoricalStreamArchiveManager
from storage import SQLiteHistoryStore


NOW = datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)


def source_registry() -> LongVideoSourceRegistry:
    return LongVideoSourceRegistry.from_dict(
        {
            "schema_version": 2,
            "timezone": "Asia/Shanghai",
            "sources": [
                {
                    "source_id": "main-monitor",
                    "display_name": "皮带主监控",
                    "source_kind": "rtsp",
                    "video_path": "",
                    "started_at": None,
                    "line_id": "main-line",
                    "zones": [],
                    "manifest_path": "",
                    "resolution": None,
                    "duration_seconds": None,
                    "segments": [],
                    "stream": {
                        "url_env": "MAIN_MONITOR_RTSP_URL",
                        "transport": "tcp",
                        "segment_seconds": 60,
                    },
                }
            ],
        }
    )


class HistoricalStreamArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.recording_root = self.root / "recordings"
        self.recording_root.mkdir()
        self.store = SQLiteHistoryStore(
            self.root / "history.sqlite3",
            now=lambda: NOW,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def manager(self, capture_runner, *, recover_orphans=False):
        return HistoricalStreamArchiveManager(
            self.store,
            capture_runner,
            output_root=self.root / "manifests",
            allowed_recording_roots=(self.recording_root,),
            now=lambda: NOW,
            retry_seconds=0.01,
            recover_orphans=recover_orphans,
        )

    def record_segment(self, start: datetime, end: datetime, name: str):
        path = self.recording_root / name
        path.write_bytes(b"video")
        return self.store.record_stream_archive_segment(
            "main-monitor",
            started_at=start,
            ended_at=end,
            status="ready",
            video_path=str(path),
            duration_seconds=(end - start).total_seconds(),
        )

    def test_background_archive_persists_segment_manifest_and_stops(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        video_path = self.recording_root / "segment.mp4"

        def capture_runner(session_id, arguments):
            self.assertEqual(session_id, "archive")
            self.assertEqual(arguments["source_id"], "main-monitor")
            entered.set()
            release.wait(2)
            video_path.write_bytes(b"video")
            return {
                "ok": True,
                "data": {
                    "started_at": "2026-07-21T11:59:00+00:00",
                    "ended_at": "2026-07-21T12:00:00+00:00",
                    "duration_seconds": 60,
                    "video_path": str(video_path),
                    "frame_count": 1500,
                    "rtsp_url": "rtsp://user:secret@example.invalid/live",
                },
            }

        manager = self.manager(capture_runner)
        manager.start(
            "main-monitor",
            segment_seconds=60,
            retention_seconds=86400,
        )
        self.assertTrue(entered.wait(1))
        requested = manager.stop("main-monitor")
        self.assertEqual(requested.status, "stopping")
        release.set()
        terminal = manager.wait_for_terminal("main-monitor", timeout_seconds=2)

        segments = self.store.list_stream_archive_segments(
            "main-monitor", statuses=("ready",)
        )
        manifest = manager.manifest_path("main-monitor")
        self.assertEqual(terminal.status, "stopped")
        self.assertEqual(len(segments), 1)
        self.assertTrue(manifest.is_file())
        manifest_text = manifest.read_text(encoding="utf-8")
        self.assertNotIn("rtsp://", manifest_text)
        self.assertNotIn("secret", manifest_text)

    def test_resolve_range_accepts_small_boundary_gap_and_reports_real_gap(self) -> None:
        self.record_segment(NOW - timedelta(hours=2), NOW - timedelta(hours=1, minutes=30), "a.mp4")
        self.record_segment(
            NOW - timedelta(hours=1, minutes=29, seconds=59),
            NOW - timedelta(hours=1),
            "b.mp4",
        )
        manager = self.manager(lambda *_: {})
        complete = manager.resolve_range(
            "main-monitor",
            start_time=NOW - timedelta(hours=2),
            end_time=NOW - timedelta(hours=1),
            tolerance_seconds=2,
        )
        missing = manager.resolve_range(
            "main-monitor",
            start_time=NOW - timedelta(hours=3),
            end_time=NOW - timedelta(hours=1),
            tolerance_seconds=2,
        )

        self.assertTrue(complete.complete)
        self.assertFalse(missing.complete)
        self.assertEqual(
            missing.gaps[0]["start_time"],
            (NOW - timedelta(hours=3)).isoformat(timespec="seconds"),
        )

    def test_resolve_range_rejects_indexed_segment_when_file_is_missing(self) -> None:
        self.store.record_stream_archive_segment(
            "main-monitor",
            started_at=NOW - timedelta(hours=1),
            ended_at=NOW - timedelta(minutes=30),
            status="ready",
            video_path=str(self.recording_root / "missing.mp4"),
            duration_seconds=1800,
        )
        result = self.manager(lambda *_: {}).resolve_range(
            "main-monitor",
            start_time=NOW - timedelta(hours=1),
            end_time=NOW - timedelta(minutes=30),
        )
        self.assertFalse(result.complete)
        self.assertEqual(len(result.missing_segments), 1)

    def test_cleanup_deletes_only_expired_files_inside_allowed_root(self) -> None:
        self.store.upsert_stream_archive_state(
            "main-monitor",
            status="stopped",
            segment_seconds=60,
            retention_seconds=86400,
        )
        old = self.record_segment(NOW - timedelta(days=2), NOW - timedelta(days=2, minutes=-1), "old.mp4")
        recent = self.record_segment(NOW - timedelta(hours=1), NOW - timedelta(minutes=59), "recent.mp4")
        outside = self.root / "outside.mp4"
        outside.write_bytes(b"do-not-delete")
        unsafe = self.store.record_stream_archive_segment(
            "main-monitor",
            started_at=NOW - timedelta(days=3),
            ended_at=NOW - timedelta(days=3, minutes=-1),
            status="ready",
            video_path=str(outside),
            duration_seconds=60,
        )

        result = self.manager(lambda *_: {}).cleanup_expired("main-monitor")

        self.assertIn(old.segment_id, result["deleted_segment_ids"])
        self.assertIn(unsafe.segment_id, result["refused_segment_ids"])
        self.assertFalse(Path(old.video_path).exists())
        self.assertTrue(Path(recent.video_path).exists())
        self.assertTrue(outside.exists())

    def test_restart_marks_active_archive_failed_without_auto_resume(self) -> None:
        self.store.upsert_stream_archive_state(
            "main-monitor",
            status="running",
            segment_seconds=60,
            retention_seconds=86400,
        )
        self.manager(lambda *_: {}, recover_orphans=True)
        state = self.store.get_stream_archive_state("main-monitor")
        self.assertEqual(state.status, "failed")
        self.assertIn("重启", state.last_error)

    def test_historical_detection_clips_boundaries_and_reuses_video_pipeline(self) -> None:
        archive_start = NOW - timedelta(hours=2)
        archive_end = NOW - timedelta(hours=1)
        segment = self.record_segment(archive_start, archive_end, "hour.mp4")
        manager = self.manager(lambda *_: {})
        segment_calls = []
        detection_calls = []

        def video_segmenter(path, start_offset, end_offset):
            segment_calls.append((path, start_offset, end_offset))
            clipped = self.recording_root / "clipped.mp4"
            clipped.write_bytes(b"clip")
            return clipped

        def detection_runner(path, video_start, parameters):
            detection_calls.append((path, video_start, parameters))
            return VideoDetectionOutcome(
                {
                    "status": "completed",
                    "video_start_time": video_start.isoformat(),
                    "video_end_time": (video_start + timedelta(minutes=10)).isoformat(),
                    "duration_seconds": 600,
                    "num_events": 1,
                    "class_counts": {"石块异物": 1},
                    "events": [{"event_id": 1, "key_frame": "outputs/frame.jpg"}],
                },
                {
                    "report_id": "alarm-history",
                    "overall_risk": {"level": "high", "requires_stop": True},
                },
                "历史录像报警报告",
            )

        tools = AgentTools(
            self.store,
            detection_runner=detection_runner,
            video_segmenter=video_segmenter,
            video_source_registry_loader=source_registry,
            archive_manager=manager,
            now=lambda: NOW,
        )
        service = AgentService(self.store, tools=tools)
        result = service.run_skill(
            "detect-archived-video",
            session_id="operator",
            arguments={
                "source_id": "main-monitor",
                "start_time": (archive_start + timedelta(minutes=10)).isoformat(),
                "end_time": (archive_start + timedelta(minutes=20)).isoformat(),
                "parameters": {"sample_fps": 1.0},
            },
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["event_count"], 1)
        self.assertEqual(result["data"]["alarm_report"], "历史录像报警报告")
        self.assertEqual(segment_calls[0][1:], (600.0, 1200.0))
        self.assertEqual(detection_calls[0][1], archive_start + timedelta(minutes=10))
        self.assertEqual(result["data"]["segments"][0]["segment_id"], segment.segment_id)

    def test_historical_detection_refuses_coverage_gap_without_running_model(self) -> None:
        calls = []

        def detection_runner(*args):
            calls.append(args)
            raise AssertionError("coverage gap must not run inference")

        manager = self.manager(lambda *_: {})
        tools = AgentTools(
            self.store,
            detection_runner=detection_runner,
            video_source_registry_loader=source_registry,
            archive_manager=manager,
            now=lambda: NOW,
        )
        result = AgentService(self.store, tools=tools).run_skill(
            "detect-archived-video",
            arguments={
                "source_id": "main-monitor",
                "start_time": (NOW - timedelta(hours=1)).isoformat(),
                "end_time": (NOW - timedelta(minutes=30)).isoformat(),
            },
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "archive_coverage_gap")
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
