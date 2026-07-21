from __future__ import annotations

import tempfile
import unittest
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from agent import (
    AgentService,
    AgentTools,
    ImageDetectionOutcome,
    LongVideoSourceRegistry,
    StreamCaptureResult,
    StreamProbeResult,
    VideoDetectionOutcome,
)
from storage import SQLiteHistoryStore


FIXED_NOW = datetime(2026, 7, 16, 10, 30, tzinfo=timezone.utc)


class AgentSkillTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.store = SQLiteHistoryStore(
            self.root / "history.sqlite3", now=lambda: FIXED_NOW
        )
        self.video_calls = []
        self.image_calls = []
        self.segment_calls = []
        self.probe_calls = []
        self.capture_calls = []

        def video_runner(video_path, video_start, parameters):
            self.video_calls.append((video_path, video_start, parameters))
            detection = {
                "status": "completed",
                "video": str(video_path),
                "video_start_time": video_start.isoformat(),
                "video_end_time": "2026-07-16T08:30:00+00:00",
                "duration_seconds": 1800,
                "num_events": 1,
                "class_counts": {"石块异物": 1},
                "events": [
                    {
                        "event_id": 1,
                        "key_frame": "outputs/test-event-frame.jpg",
                    }
                ],
            }
            alarm = {
                "report_id": f"alarm-{video_path.stem}",
                "overall_risk": {"level": "medium", "requires_stop": False},
            }
            return VideoDetectionOutcome(detection, alarm, "视频报警报告")

        def image_runner(image_path, parameters):
            self.image_calls.append((image_path, parameters))
            detection = {
                "status": "detected",
                "source": str(image_path),
                "num_images": 1,
                "num_detections": 1,
                "num_candidates": 0,
                "has_foreign_object": True,
                "class_counts": {"金属异物": 1},
                "objects": [{"class": "metal"}],
            }
            alarm = {
                "report_id": f"alarm-{image_path.stem}",
                "overall_risk": {"level": "high", "requires_stop": True},
            }
            return ImageDetectionOutcome(detection, alarm, "图片报警报告")

        def video_segmenter(video_path, start_offset, end_offset):
            self.segment_calls.append((video_path, start_offset, end_offset))
            segment_path = self.root / "segment.mp4"
            segment_path.write_bytes(b"segment")
            return segment_path

        source_registry = LongVideoSourceRegistry.from_dict(
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
                        "zones": [
                            {
                                "zone_id": "belt-zone-a",
                                "display_name": "皮带A区",
                                "roi": [10, 20, 300, 320],
                            }
                        ],
                        "manifest_path": "",
                        "resolution": None,
                        "duration_seconds": None,
                        "segments": [],
                        "stream": {
                            "url_env": "MAIN_MONITOR_RTSP_URL",
                            "transport": "tcp",
                        },
                    }
                ],
            }
        )

        def stream_probe_runner(source, environment):
            self.probe_calls.append((source, environment))
            return StreamProbeResult(
                source_id=source.source_id,
                display_name=source.display_name,
                line_id=source.line_id,
                online=True,
                checked_at=FIXED_NOW.isoformat(),
                latency_ms=120,
                width=640,
                height=360,
                fps=25.0,
                codec="h264",
                backend="FFMPEG",
                transport="tcp",
            )

        def stream_capture_runner(source, duration_seconds, environment):
            self.capture_calls.append((source, duration_seconds, environment))
            capture_path = self.root / "capture.mp4"
            metadata_path = self.root / "capture.json"
            capture_path.write_bytes(b"capture")
            metadata_path.write_text("{}", encoding="utf-8")
            return StreamCaptureResult(
                source_id=source.source_id,
                display_name=source.display_name,
                line_id=source.line_id,
                captured=True,
                requested_duration_seconds=duration_seconds or 60.0,
                started_at=FIXED_NOW.isoformat(),
                ended_at=FIXED_NOW.isoformat(),
                duration_seconds=3.0,
                frame_count=75,
                width=640,
                height=360,
                fps=25.0,
                source_codec="h264",
                output_codec="mp4v",
                backend="FFMPEG",
                transport="tcp",
                video_path=str(capture_path),
                metadata_path=str(metadata_path),
            )

        tools = AgentTools(
            self.store,
            detection_runner=video_runner,
            image_detection_runner=image_runner,
            video_segmenter=video_segmenter,
            stream_probe_runner=stream_probe_runner,
            stream_capture_runner=stream_capture_runner,
            video_source_registry_loader=lambda: source_registry,
            now=lambda: FIXED_NOW,
        )
        self.service = AgentService(self.store, tools=tools)
        self.video = self.root / "belt.mp4"
        self.video.write_bytes(b"video")
        self.image = self.root / "belt.jpg"
        self.image.write_bytes(b"image")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_catalog_exposes_closed_skill_set(self) -> None:
        catalog = self.service.skill_catalog()
        self.assertEqual(
            {item["name"] for item in catalog},
            {
                "detect-image",
                "detect-video",
                "assess-risk",
                "parse-detection-result",
                "control-alarm",
                "query-history",
                "generate-risk-report",
                "review-detection",
                "run-inspection-task",
                "probe-video-source",
                "capture-video-source",
                "detect-video-source",
            },
        )
        alarm_spec = next(item for item in catalog if item["name"] == "control-alarm")
        action_schema = alarm_spec["input_schema"]["properties"]["action"]
        self.assertEqual(action_schema["enum"], ["query", "confirm", "cancel"])
        self.assertEqual(action_schema["default"], "query")
        self.assertEqual(action_schema["aliases"]["view"], "query")
        for item in catalog:
            self.assertIn("input_schema", item)
            self.assertFalse(item["input_schema"]["additionalProperties"])
            self.assertEqual(
                set(item["input_schema"]["properties"]),
                set(item["required_inputs"]) | set(item["optional_inputs"]),
            )
        with self.assertRaises(LookupError):
            self.service.run_skill("arbitrary-python", arguments={})

    def test_probe_video_source_uses_registered_source_and_strict_arguments(self) -> None:
        result = self.service.run_skill(
            "probe-video-source",
            session_id="operator",
            arguments={"source_id": " MAIN-MONITOR "},
        )

        self.assertTrue(result["ok"])
        self.assertTrue(result["data"]["online"])
        self.assertEqual(result["data"]["source_id"], "main-monitor")
        self.assertEqual(result["data"]["codec"], "h264")
        self.assertEqual(self.probe_calls[0][0].source_id, "main-monitor")
        self.assertIsNone(self.probe_calls[0][1])

        for arguments in (
            {},
            {"source_id": "rtsp://camera/live"},
            {"source_id": "main-monitor", "transport": "udp"},
        ):
            invalid = self.service.run_skill(
                "probe-video-source",
                arguments=arguments,
            )
            self.assertFalse(invalid["ok"])
            self.assertEqual(invalid["error_code"], "invalid_arguments")

    def test_capture_video_source_uses_registered_source_and_bounded_duration(self) -> None:
        result = self.service.run_skill(
            "capture-video-source",
            session_id="operator",
            arguments={"source_id": "main-monitor", "duration_seconds": 3},
        )

        self.assertTrue(result["ok"])
        self.assertTrue(result["data"]["captured"])
        self.assertEqual(result["data"]["frame_count"], 75)
        self.assertEqual(self.capture_calls[0][0].source_id, "main-monitor")
        self.assertEqual(self.capture_calls[0][1], 3.0)
        self.assertIsNone(self.capture_calls[0][2])

        for arguments in (
            {"source_id": "main-monitor", "duration_seconds": 0},
            {"source_id": "main-monitor", "duration_seconds": 3601},
            {"source_id": "main-monitor", "output_path": "capture.mp4"},
            {"source_id": "main-monitor", "rtsp_url": "rtsp://camera/live"},
        ):
            invalid = self.service.run_skill(
                "capture-video-source",
                arguments=arguments,
            )
            self.assertFalse(invalid["ok"])
            self.assertEqual(invalid["error_code"], "invalid_arguments")

    def test_detect_video_source_runs_capture_detection_risk_and_history_workflow(self) -> None:
        result = self.service.run_skill(
            "detect-video-source",
            session_id="operator",
            arguments={
                "source_id": "main-monitor",
                "duration_seconds": 3,
                "zone_id": "belt-zone-a",
                "parameters": {
                    "sample_fps": 2.0,
                    "conf": 0.2,
                    "known_conf": 0.5,
                },
            },
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["source_id"], "main-monitor")
        self.assertEqual(result["data"]["line_id"], "main-line")
        self.assertEqual(result["data"]["zone"]["zone_id"], "belt-zone-a")
        self.assertEqual(result["data"]["zone"]["roi"], [10, 20, 300, 320])
        self.assertEqual(result["data"]["capture"]["frame_count"], 75)
        self.assertEqual(result["data"]["alarm_report"], "视频报警报告")
        self.assertEqual(
            result["data"]["event_frames"],
            [{"event_id": 1, "key_frame": "outputs/test-event-frame.jpg"}],
        )
        self.assertEqual(
            result["data"]["workflow"],
            [
                "capture-video-source",
                "detect-video",
                "assess-risk",
                "persist-history",
                "create-alarm",
            ],
        )
        self.assertEqual(self.video_calls[-1][0], self.root / "capture.mp4")
        self.assertEqual(self.video_calls[-1][1], FIXED_NOW)
        self.assertEqual(self.video_calls[-1][2]["sample_fps"], 2.0)
        self.assertEqual(self.video_calls[-1][2]["roi"], (10, 20, 300, 320))
        record = self.store.latest_detection("operator")
        self.assertEqual(record.line_id, "main-line")
        self.assertEqual(record.source_started_at, FIXED_NOW.isoformat())

    def test_detect_video_source_rejects_unknown_zone_and_conflicting_roi(self) -> None:
        unknown_zone = self.service.run_skill(
            "detect-video-source",
            arguments={"source_id": "main-monitor", "zone_id": "missing-zone"},
        )
        conflicting_roi = self.service.run_skill(
            "detect-video-source",
            arguments={
                "source_id": "main-monitor",
                "zone_id": "belt-zone-a",
                "parameters": {"roi": [0, 0, 100, 100]},
            },
        )

        self.assertFalse(unknown_zone["ok"])
        self.assertEqual(unknown_zone["error_code"], "zone_not_found")
        self.assertEqual(unknown_zone["data"]["available_zones"], ["belt-zone-a"])
        self.assertFalse(conflicting_roi["ok"])
        self.assertEqual(conflicting_roi["error_code"], "invalid_arguments")

    def test_alarm_view_alias_is_a_safe_read_only_fallback(self) -> None:
        result = self.service.run_skill(
            "control-alarm",
            session_id="operator",
            arguments={"action": "view"},
        )

        self.assertTrue(result["ok"])
        self.assertFalse(result["data"]["found"])

        invalid = self.service.run_skill(
            "control-alarm",
            session_id="operator",
            arguments={"action": "delete"},
        )
        self.assertFalse(invalid["ok"])
        self.assertEqual(invalid["error_code"], "invalid_arguments")

    def test_existing_database_is_migrated_without_losing_rows(self) -> None:
        legacy_path = self.root / "legacy.sqlite3"
        connection = sqlite3.connect(legacy_path)
        connection.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE detection_runs (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                source_type TEXT NOT NULL,
                source_path TEXT NOT NULL,
                status TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                alarm_report TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            INSERT INTO sessions VALUES ('legacy', '2026-07-15T08:00:00', '2026-07-15T08:00:00');
            INSERT INTO detection_runs VALUES (
                'det-legacy', 'legacy', 'image', 'legacy.jpg', 'completed',
                'low', '{}', '', '2026-07-15T08:00:00'
            );
            """
        )
        connection.commit()
        connection.close()

        migrated = SQLiteHistoryStore(legacy_path, now=lambda: FIXED_NOW)
        record = migrated.get_detection("det-legacy")

        self.assertIsNotNone(record)
        self.assertEqual(record.line_id, "")
        self.assertEqual(record.review_status, "unreviewed")

    def test_video_parameters_roi_and_metadata_are_forwarded(self) -> None:
        result = self.service.run_skill(
            "detect-video",
            session_id="operator",
            arguments={
                "video_path": str(self.video),
                "video_start_time": "2026-07-16T08:00:00+00:00",
                "line_id": "line-1",
                "parameters": {
                    "sample_fps": 2.5,
                    "conf": 0.2,
                    "known_conf": 0.5,
                    "roi": [10, 20, 300, 400],
                },
            },
        )

        self.assertTrue(result["ok"])
        self.assertEqual(self.video_calls[0][2]["sample_fps"], 2.5)
        self.assertEqual(self.video_calls[0][2]["roi"], (10, 20, 300, 400))
        record = self.store.latest_detection("operator")
        self.assertEqual(record.line_id, "line-1")
        self.assertEqual(record.source_started_at, "2026-07-16T08:00:00+00:00")
        self.assertEqual(record.source_ended_at, "2026-07-16T08:30:00+00:00")

    def test_video_schema_applies_defaults_and_rejects_nested_type_errors(self) -> None:
        result = self.service.run_skill(
            "detect-video",
            arguments={
                "video_path": str(self.video),
                "parameters": {"conf": 0.2, "known_conf": 0.5},
            },
        )

        self.assertTrue(result["ok"])
        parameters = self.video_calls[-1][2]
        self.assertEqual(parameters["sample_fps"], 4.0)
        self.assertEqual(parameters["imgsz"], 800)
        self.assertEqual(parameters["track_center_distance_ratio"], 3.0)
        self.assertIsNone(parameters["roi"])

        invalid_type = self.service.run_skill(
            "detect-video",
            arguments={
                "video_path": str(self.video),
                "parameters": {"sample_fps": "2"},
            },
        )
        unknown_nested = self.service.run_skill(
            "detect-video",
            arguments={
                "video_path": str(self.video),
                "parameters": {"unsupported_threshold": 0.5},
            },
        )

        self.assertFalse(invalid_type["ok"])
        self.assertEqual(invalid_type["error_code"], "invalid_arguments")
        self.assertFalse(unknown_nested["ok"])
        self.assertEqual(unknown_nested["error_code"], "invalid_arguments")

    def test_video_offsets_use_segment_adapter_and_adjust_real_start_time(self) -> None:
        result = self.service.run_skill(
            "detect-video",
            session_id="operator",
            arguments={
                "video_path": str(self.video),
                "video_start_time": "2026-07-16T08:00:00+00:00",
                "start_offset_seconds": 60,
                "end_offset_seconds": 120,
            },
        )

        self.assertTrue(result["ok"])
        self.assertEqual(self.segment_calls[0], (self.video, 60.0, 120.0))
        self.assertEqual(self.video_calls[-1][0], self.root / "segment.mp4")
        self.assertEqual(
            self.video_calls[-1][1].isoformat(),
            "2026-07-16T08:01:00+00:00",
        )
        self.assertEqual(result["data"]["start_offset_seconds"], 60.0)
        self.assertEqual(result["data"]["end_offset_seconds"], 120.0)

        invalid = self.service.run_skill(
            "detect-video",
            arguments={
                "video_path": str(self.video),
                "start_offset_seconds": 120,
                "end_offset_seconds": 60,
            },
        )
        self.assertFalse(invalid["ok"])
        self.assertEqual(invalid["error_code"], "invalid_arguments")

    def test_invalid_parameters_fail_before_detector_call(self) -> None:
        result = self.service.run_skill(
            "detect-video",
            arguments={
                "video_path": str(self.video),
                "parameters": {"sample_fps": 0, "roi": [10, 20, 5, 30]},
            },
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "invalid_arguments")
        self.assertEqual(self.video_calls, [])

    def test_history_filters_report_and_review_closed_loop(self) -> None:
        detected = self.service.run_skill(
            "detect-image",
            session_id="operator",
            arguments={
                "image_path": str(self.image),
                "line_id": "line-2",
                "captured_at": "2026-07-16T09:10:00+00:00",
            },
        )
        detection_id = detected["data"]["detection_id"]
        reviewed = self.service.run_skill(
            "review-detection",
            session_id="operator",
            arguments={
                "detection_id": detection_id,
                "action": "close",
                "reviewer": "张工",
                "note": "异物已清除",
            },
        )
        history = self.service.run_skill(
            "query-history",
            arguments={
                "start_time": "2026-07-16T09:00:00+00:00",
                "end_time": "2026-07-16T10:00:00+00:00",
                "risk_level": "high",
                "line_id": "line-2",
                "review_status": "closed",
            },
        )
        report = self.service.run_skill(
            "generate-risk-report",
            arguments={"date": "2026-07-16", "line_id": "line-2"},
        )

        self.assertTrue(reviewed["ok"])
        self.assertEqual(history["data"]["count"], 1)
        self.assertEqual(history["data"]["records"][0]["reviewer"], "张工")
        self.assertEqual(report["data"]["detection_count"], 1)
        self.assertEqual(report["data"]["class_counts"], {"金属异物": 1})
        actions = self.store.list_detection_review_actions(detection_id)
        self.assertEqual(actions[0]["action"], "close")

    def test_history_date_and_filter_aliases_are_normalized(self) -> None:
        self.service.run_skill(
            "detect-image",
            arguments={
                "image_path": str(self.image),
                "line_id": "line-2",
                "captured_at": "2026-07-16T09:10:00+00:00",
            },
        )

        history = self.service.run_skill(
            "query-history",
            arguments={
                "date": "2026-07-16",
                "risk_level": "高风险",
                "source_type": "图片",
                "line_id": "line-2",
            },
        )
        conflict = self.service.run_skill(
            "query-history",
            arguments={
                "date": "2026-07-16",
                "start_time": "2026-07-16T08:00:00+00:00",
            },
        )

        self.assertTrue(history["ok"])
        self.assertEqual(history["data"]["count"], 1)
        self.assertFalse(conflict["ok"])
        self.assertEqual(conflict["error_code"], "invalid_arguments")

    def test_review_action_is_required_and_closed_enum(self) -> None:
        missing = self.service.run_skill("review-detection", arguments={})
        invalid = self.service.run_skill(
            "review-detection",
            arguments={"action": "approve"},
        )

        self.assertFalse(missing["ok"])
        self.assertEqual(missing["error_code"], "invalid_arguments")
        self.assertFalse(invalid["ok"])
        self.assertEqual(invalid["error_code"], "invalid_arguments")

    def test_composite_inspection_infers_image_and_persists_alarm(self) -> None:
        result = self.service.run_skill(
            "run-inspection-task",
            session_id="operator",
            arguments={"media_path": str(self.image), "line_id": "line-3"},
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["workflow"][0], "detect-image")
        current = self.service.run_skill(
            "control-alarm", arguments={"action": "query", "line_id": "line-3"}
        )
        self.assertTrue(current["data"]["found"])
        self.assertEqual(current["data"]["risk_level"], "high")

    def test_composite_inspection_accepts_explicit_path_and_rejects_conflicts(self) -> None:
        explicit_path = self.service.run_skill(
            "run-inspection-task",
            session_id="operator",
            arguments={"image_path": str(self.image), "line_id": "line-3"},
        )
        conflicting = self.service.run_skill(
            "run-inspection-task",
            arguments={
                "image_path": str(self.image),
                "video_path": str(self.video),
            },
        )
        self.assertTrue(explicit_path["ok"])
        self.assertFalse(conflicting["ok"])

    def test_risk_skill_reads_detection_object(self) -> None:
        detection = {
            "status": "detected",
            "timestamp": "2026-07-16 09:00:00",
            "source": "sample.jpg",
            "num_images": 1,
            "num_detections": 1,
            "has_foreign_object": True,
            "objects": [
                {
                    "image": "sample.jpg",
                    "class_id": 2,
                    "class": "metal",
                    "class_name": "金属异物",
                    "confidence": 0.91,
                    "bbox_xyxy": [10, 10, 100, 100],
                }
            ],
        }
        result = self.service.run_skill(
            "assess-risk", arguments={"detection": detection, "source_type": "image"}
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["risk_level"], "high")
        self.assertTrue(result["data"]["requires_stop"])

        parsed = self.service.run_skill(
            "parse-detection-result",
            arguments={"detection": detection, "source_type": "image"},
        )
        self.assertTrue(parsed["ok"])
        self.assertEqual(parsed["data"]["event_count"], 1)
        object_result = parsed["data"]["events"][0]["objects"][0]
        self.assertEqual(object_result["confidence"], 0.91)
        self.assertEqual(object_result["bbox_xyxy"], [10.0, 10.0, 100.0, 100.0])


if __name__ == "__main__":
    unittest.main()
