from __future__ import annotations

import json
import os
import unittest
from datetime import datetime, timezone

from agent.streaming import RtspStreamProbe
from agent.video_sources import LongVideoSource


class _Frame:
    shape = (360, 640, 3)


class _FakeCv2:
    CAP_FFMPEG = 1900
    CAP_PROP_OPEN_TIMEOUT_MSEC = 53
    CAP_PROP_READ_TIMEOUT_MSEC = 54
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_FOURCC = 6


class _FakeCapture:
    def __init__(self, *, opened=True, frame_ok=True, frame=None):
        self.opened = opened
        self.frame_ok = frame_ok
        self.frame = _Frame() if frame is None and frame_ok else frame
        self.released = False
        self.open_call = None

    def open(self, url, backend, parameters):
        self.open_call = (url, backend, list(parameters))
        return self.opened

    def isOpened(self):
        return self.opened

    def read(self):
        return self.frame_ok, self.frame

    def get(self, property_id):
        values = {
            _FakeCv2.CAP_PROP_FRAME_WIDTH: 640,
            _FakeCv2.CAP_PROP_FRAME_HEIGHT: 360,
            _FakeCv2.CAP_PROP_FPS: 25.0,
            _FakeCv2.CAP_PROP_FOURCC: sum(
                ord(character) << (8 * index)
                for index, character in enumerate("avc1")
            ),
        }
        return values.get(property_id, 0)

    def getBackendName(self):
        return "FFMPEG"

    def release(self):
        self.released = True


def _source(*, source_kind="rtsp"):
    raw = {
        "source_id": "main-monitor",
        "display_name": "皮带主监控",
        "source_kind": source_kind,
        "video_path": "" if source_kind == "rtsp" else "data/archive.mp4",
        "started_at": None,
        "line_id": "main-line",
        "zones": [],
        "manifest_path": "",
        "resolution": None,
        "duration_seconds": None,
        "segments": [],
        "stream": (
            {
                "url_env": "MAIN_MONITOR_RTSP_URL",
                "transport": "tcp",
                "connect_timeout_seconds": 2.0,
                "read_timeout_seconds": 3.0,
            }
            if source_kind == "rtsp"
            else None
        ),
    }
    return LongVideoSource.from_dict(raw, timezone_name="Asia/Shanghai")


class StreamProbeTests(unittest.TestCase):
    def setUp(self):
        self.now = lambda: datetime(2026, 7, 20, 8, 0, tzinfo=timezone.utc)

    @staticmethod
    def _clock(step=0.05):
        current = [0.0]

        def monotonic():
            value = current[0]
            current[0] += step
            return value

        return monotonic

    def test_online_probe_returns_metadata_and_releases_capture(self):
        capture = _FakeCapture()
        original_options = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        probe = RtspStreamProbe(
            capture_factory=lambda: capture,
            cv2_module=_FakeCv2,
            now=self.now,
            monotonic=self._clock(),
        )
        secret_url = "rtsp://operator:super-secret@127.0.0.1:8554/main-monitor"

        result = probe(
            _source(),
            {"MAIN_MONITOR_RTSP_URL": secret_url},
        )

        self.assertTrue(result.online)
        self.assertEqual((result.width, result.height), (640, 360))
        self.assertEqual(result.fps, 25.0)
        self.assertEqual(result.codec, "h264")
        self.assertEqual(result.backend, "FFMPEG")
        self.assertEqual(result.transport, "tcp")
        self.assertTrue(capture.released)
        self.assertEqual(capture.open_call[0], secret_url)
        self.assertIn(_FakeCv2.CAP_PROP_OPEN_TIMEOUT_MSEC, capture.open_call[2])
        self.assertEqual(
            os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS"),
            original_options,
        )
        serialized = json.dumps(result.to_dict(), ensure_ascii=False)
        self.assertNotIn(secret_url, serialized)
        self.assertNotIn("operator", serialized)
        self.assertNotIn("super-secret", serialized)

    def test_missing_environment_variable_returns_safe_configuration_error(self):
        capture = _FakeCapture()
        result = RtspStreamProbe(
            capture_factory=lambda: capture,
            cv2_module=_FakeCv2,
            now=self.now,
            monotonic=self._clock(),
        )(_source(), {})

        self.assertFalse(result.online)
        self.assertEqual(result.error_code, "configuration_error")
        self.assertIn("MAIN_MONITOR_RTSP_URL", result.error_message)
        self.assertIsNone(capture.open_call)
        self.assertFalse(capture.released)

    def test_connection_failure_and_empty_frame_are_safe_and_release_capture(self):
        secret_url = "rtsp://user:password@camera.local/live"
        failed_capture = _FakeCapture(opened=False)
        failed = RtspStreamProbe(
            capture_factory=lambda: failed_capture,
            cv2_module=_FakeCv2,
            now=self.now,
            monotonic=self._clock(),
        )(_source(), {"MAIN_MONITOR_RTSP_URL": secret_url})
        self.assertFalse(failed.online)
        self.assertEqual(failed.error_code, "connection_failed")
        self.assertTrue(failed_capture.released)

        empty_capture = _FakeCapture(frame_ok=False)
        empty = RtspStreamProbe(
            capture_factory=lambda: empty_capture,
            cv2_module=_FakeCv2,
            now=self.now,
            monotonic=self._clock(),
        )(_source(), {"MAIN_MONITOR_RTSP_URL": secret_url})
        self.assertFalse(empty.online)
        self.assertEqual(empty.error_code, "no_video_frame")
        self.assertTrue(empty_capture.released)

        combined = json.dumps(
            {"failed": failed.to_dict(), "empty": empty.to_dict()},
            ensure_ascii=False,
        )
        self.assertNotIn(secret_url, combined)
        self.assertNotIn("password", combined)

    def test_non_rtsp_source_is_rejected_without_opening_capture(self):
        capture = _FakeCapture()
        result = RtspStreamProbe(
            capture_factory=lambda: capture,
            cv2_module=_FakeCv2,
            now=self.now,
            monotonic=self._clock(),
        )(_source(source_kind="file"), {})

        self.assertFalse(result.online)
        self.assertEqual(result.error_code, "not_rtsp_source")
        self.assertIsNone(capture.open_call)


if __name__ == "__main__":
    unittest.main()
