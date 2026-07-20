from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent.video_sources import LongVideoSourceRegistry, load_video_source_registry
from project_config import VIDEO_SOURCES_PATH


def _registry(source: dict, *, schema_version: int = 2) -> dict:
    return {
        "schema_version": schema_version,
        "timezone": "Asia/Shanghai",
        "sources": [source],
    }


def _file_source(**overrides) -> dict:
    source = {
        "source_id": "main-monitor",
        "display_name": "皮带主监控",
        "source_kind": "file",
        "video_path": "data/monitor/main.mp4",
        "started_at": "2026-07-20T08:00:00+08:00",
        "line_id": "main-line",
        "zones": [],
        "manifest_path": "data/monitor/manifest.json",
        "resolution": {"width": 1920, "height": 1080},
        "duration_seconds": 7200,
        "segments": [],
        "stream": None,
    }
    source.update(overrides)
    return source


def _rtsp_source(**overrides) -> dict:
    source = {
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
            "capture_window_seconds": 60,
            "segment_seconds": 60,
            "reconnect_seconds": 5.0,
            "connect_timeout_seconds": 10.0,
            "read_timeout_seconds": 15.0,
        },
    }
    source.update(overrides)
    return source


class LongVideoSourceTests(unittest.TestCase):
    def test_default_source_is_rtsp_and_uses_environment_reference(self) -> None:
        registry = load_video_source_registry()
        source = registry.get("main-monitor")

        self.assertEqual(VIDEO_SOURCES_PATH.name, "video_sources.json")
        self.assertEqual(registry.schema_version, 2)
        self.assertEqual(registry.timezone, "Asia/Shanghai")
        self.assertEqual(source.display_name, "皮带主监控")
        self.assertTrue(source.is_rtsp)
        self.assertEqual(source.video_path, "")
        self.assertIsNone(source.started_at)
        self.assertIsNone(source.duration_seconds)
        self.assertEqual(source.stream.url_env, "MAIN_MONITOR_RTSP_URL")
        self.assertEqual(source.stream.transport, "tcp")
        self.assertIn(
            "MAIN_MONITOR_RTSP_URL 未配置",
            source.readiness_issues(environment={}),
        )
        self.assertEqual(
            source.readiness_issues(
                environment={
                    "MAIN_MONITOR_RTSP_URL": "rtsp://127.0.0.1:8554/main-monitor"
                }
            ),
            (),
        )

    def test_rtsp_url_is_resolved_without_entering_serialized_config(self) -> None:
        registry = LongVideoSourceRegistry.from_dict(_registry(_rtsp_source()))
        source = registry.get("main-monitor")
        url = source.resolve_stream_url(
            {"MAIN_MONITOR_RTSP_URL": "rtsp://camera-user:secret@127.0.0.1:8554/live"}
        )
        serialized = json.dumps(registry.to_dict(), ensure_ascii=False)

        self.assertEqual(url, "rtsp://camera-user:secret@127.0.0.1:8554/live")
        self.assertNotIn("camera-user", serialized)
        self.assertNotIn("secret", serialized)
        self.assertIn("MAIN_MONITOR_RTSP_URL", serialized)

    def test_complete_file_source_supports_segments_and_split_screen_roi(self) -> None:
        raw = _registry(
            _file_source(
                manifest_path="",
                zones=[
                    {
                        "zone_id": "area-a",
                        "display_name": "A区",
                        "roi": [0, 0, 960, 1080],
                    },
                    {
                        "zone_id": "area-b",
                        "display_name": "B区",
                        "roi": [960, 0, 1920, 1080],
                    },
                ],
                segments=[
                    {
                        "segment_id": "segment-001",
                        "video_path": "data/monitor/segment-001.mp4",
                        "started_at": "2026-07-20T08:00:00+08:00",
                        "duration_seconds": 3600,
                        "start_offset_seconds": 0,
                        "end_offset_seconds": 3600,
                    },
                    {
                        "segment_id": "segment-002",
                        "video_path": "data/monitor/segment-002.mp4",
                        "started_at": "2026-07-20T09:00:00+08:00",
                        "duration_seconds": 3600,
                        "start_offset_seconds": 3600,
                        "end_offset_seconds": 7200,
                    },
                ],
            )
        )

        registry = LongVideoSourceRegistry.from_dict(raw)
        source = registry.get("main-monitor")

        self.assertTrue(source.is_file)
        self.assertTrue(source.is_ready)
        self.assertTrue(source.has_archive)
        self.assertEqual(source.resolution.width, 1920)
        self.assertEqual(source.zones[1].roi, (960, 0, 1920, 1080))
        self.assertEqual(source.segments[1].start_offset_seconds, 3600.0)
        self.assertEqual(registry.to_dict(), raw)

    def test_schema_v1_file_source_remains_loadable_and_round_trips(self) -> None:
        source_v1 = _file_source()
        source_v1.pop("source_kind")
        source_v1.pop("stream")
        raw = _registry(source_v1, schema_version=1)

        registry = LongVideoSourceRegistry.from_dict(raw)
        source = registry.get("main-monitor")

        self.assertTrue(source.is_file)
        self.assertEqual(registry.to_dict(), raw)

    def test_registry_loads_from_an_explicit_json_file(self) -> None:
        raw = json.loads(VIDEO_SOURCES_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sources.json"
            path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

            registry = load_video_source_registry(path)

        self.assertEqual(registry.sources[0].source_id, "main-monitor")

    def test_cross_field_source_kind_rules_are_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "rtsp 视频源必须配置 stream"):
            LongVideoSourceRegistry.from_dict(
                _registry(_rtsp_source(stream=None))
            )
        with self.assertRaisesRegex(ValueError, "video_path 必须为空"):
            LongVideoSourceRegistry.from_dict(
                _registry(_rtsp_source(video_path="data/not-allowed.mp4"))
            )
        with self.assertRaisesRegex(ValueError, "duration_seconds 必须为 null"):
            LongVideoSourceRegistry.from_dict(
                _registry(_rtsp_source(duration_seconds=3600))
            )
        with self.assertRaisesRegex(ValueError, "file 视频源不能配置 stream"):
            LongVideoSourceRegistry.from_dict(
                _registry(_file_source(stream=_rtsp_source()["stream"]))
            )

    def test_stream_schema_rejects_direct_urls_invalid_types_and_unknown_fields(self) -> None:
        direct_url = _rtsp_source()
        direct_url["stream"] = {
            **direct_url["stream"],
            "url": "rtsp://camera-user:secret@example/live",
        }
        invalid_transport = _rtsp_source()
        invalid_transport["stream"] = {
            **invalid_transport["stream"],
            "transport": "view",
        }
        invalid_capture = _rtsp_source()
        invalid_capture["stream"] = {
            **invalid_capture["stream"],
            "capture_window_seconds": 0,
        }

        with self.assertRaisesRegex(ValueError, "未知字段"):
            LongVideoSourceRegistry.from_dict(_registry(direct_url))
        with self.assertRaisesRegex(ValueError, "tcp、udp 或 auto"):
            LongVideoSourceRegistry.from_dict(_registry(invalid_transport))
        with self.assertRaisesRegex(ValueError, "1 到 3600"):
            LongVideoSourceRegistry.from_dict(_registry(invalid_capture))

    def test_invalid_rtsp_environment_value_is_reported_without_secret(self) -> None:
        source = LongVideoSourceRegistry.from_dict(
            _registry(_rtsp_source())
        ).get("main-monitor")
        issues = source.readiness_issues(
            environment={"MAIN_MONITOR_RTSP_URL": "https://camera-user:secret@example/live"}
        )

        self.assertEqual(
            issues,
            ("MAIN_MONITOR_RTSP_URL 不是有效的 RTSP 地址",),
        )
        self.assertNotIn("camera-user", " ".join(issues))
        self.assertNotIn("secret", " ".join(issues))

    def test_invalid_roi_absolute_path_and_unknown_fields_are_rejected(self) -> None:
        invalid_roi = _file_source(
            zones=[
                {
                    "zone_id": "area-a",
                    "display_name": "A区",
                    "roi": [0, 0, 2000, 1080],
                }
            ]
        )
        absolute_path = _file_source(video_path=str(Path("C:/monitor/main.mp4")))
        unknown = _file_source(camera_url="rtsp://example")

        with self.assertRaisesRegex(ValueError, "不能超出"):
            LongVideoSourceRegistry.from_dict(_registry(invalid_roi))
        with self.assertRaisesRegex(ValueError, "相对于"):
            LongVideoSourceRegistry.from_dict(_registry(absolute_path))
        with self.assertRaisesRegex(ValueError, "未知字段"):
            LongVideoSourceRegistry.from_dict(_registry(unknown))


if __name__ == "__main__":
    unittest.main()
