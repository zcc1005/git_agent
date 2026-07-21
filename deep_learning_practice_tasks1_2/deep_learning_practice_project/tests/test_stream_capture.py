from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from agent.streaming import RtspStreamCapture
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

    @staticmethod
    def VideoWriter_fourcc(*characters):
        return sum(
            ord(character) << (8 * index)
            for index, character in enumerate(characters)
        )


class _FakeCapture:
    def __init__(self, *, opened=True, fail_after=None):
        self.opened = opened
        self.fail_after = fail_after
        self.read_count = 0
        self.released = False
        self.open_call = None

    def open(self, url, backend, parameters):
        self.open_call = (url, backend, list(parameters))
        return self.opened

    def isOpened(self):
        return self.opened

    def read(self):
        if self.fail_after is not None and self.read_count >= self.fail_after:
            return False, None
        self.read_count += 1
        return True, _Frame()

    def get(self, property_id):
        values = {
            _FakeCv2.CAP_PROP_FRAME_WIDTH: 640,
            _FakeCv2.CAP_PROP_FRAME_HEIGHT: 360,
            _FakeCv2.CAP_PROP_FPS: 25.0,
            _FakeCv2.CAP_PROP_FOURCC: _FakeCv2.VideoWriter_fourcc(*"avc1"),
        }
        return values.get(property_id, 0)

    def getBackendName(self):
        return "FFMPEG"

    def release(self):
        self.released = True


class _FakeWriter:
    def __init__(self, path, fourcc, fps, size, *, opened=True):
        self.path = Path(path)
        self.fourcc = fourcc
        self.fps = fps
        self.size = size
        self.opened = opened
        self.frames = []
        self.released = False

    def isOpened(self):
        return self.opened

    def write(self, frame):
        self.frames.append(frame)

    def release(self):
        if self.released:
            return
        self.released = True
        if self.opened:
            self.path.write_bytes(b"fake-mp4")


def _source():
    return LongVideoSource.from_dict(
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
                "capture_window_seconds": 60,
                "connect_timeout_seconds": 2.0,
                "read_timeout_seconds": 3.0,
            },
        },
        timezone_name="Asia/Shanghai",
    )


class StreamCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_root = Path(self.temp_dir.name)
        self.fixed_now = datetime(2026, 7, 21, 8, 0, tzinfo=timezone.utc)

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def _clock(step=0.2):
        current = [0.0]

        def monotonic():
            value = current[0]
            current[0] += step
            return value

        return monotonic

    def test_capture_writes_video_and_safe_sidecar_metadata(self):
        capture = _FakeCapture()
        writers = []

        def writer_factory(*arguments):
            writer = _FakeWriter(*arguments)
            writers.append(writer)
            return writer

        secret_url = "rtsp://operator:super-secret@127.0.0.1:8554/main-monitor"
        result = RtspStreamCapture(
            output_root=self.output_root,
            capture_factory=lambda: capture,
            writer_factory=writer_factory,
            cv2_module=_FakeCv2,
            now=lambda: self.fixed_now,
            monotonic=self._clock(),
        )(
            _source(),
            1.0,
            {"MAIN_MONITOR_RTSP_URL": secret_url},
        )

        self.assertTrue(result.captured)
        self.assertGreater(result.frame_count, 0)
        self.assertEqual(result.frame_count, 25)
        self.assertEqual(result.duration_seconds, 1.0)
        self.assertEqual((result.width, result.height), (640, 360))
        self.assertEqual(result.fps, 25.0)
        self.assertEqual(result.source_codec, "h264")
        self.assertEqual(result.output_codec, "mp4v")
        self.assertTrue(Path(result.video_path).is_file())
        self.assertTrue(Path(result.metadata_path).is_file())
        self.assertTrue(capture.released)
        self.assertTrue(writers[0].released)
        self.assertEqual(writers[0].size, (640, 360))

        serialized = json.dumps(result.to_dict(), ensure_ascii=False)
        sidecar = Path(result.metadata_path).read_text(encoding="utf-8")
        for text in (serialized, sidecar):
            self.assertNotIn(secret_url, text)
            self.assertNotIn("operator", text)
            self.assertNotIn("super-secret", text)

    def test_missing_url_fails_without_creating_files(self):
        capture = _FakeCapture()
        result = RtspStreamCapture(
            output_root=self.output_root,
            capture_factory=lambda: capture,
            writer_factory=_FakeWriter,
            cv2_module=_FakeCv2,
            now=lambda: self.fixed_now,
            monotonic=self._clock(),
        )(_source(), 1.0, {})

        self.assertFalse(result.captured)
        self.assertEqual(result.error_code, "configuration_error")
        self.assertFalse(capture.released)
        self.assertEqual(list(self.output_root.rglob("*")), [])

    def test_interrupted_stream_removes_partial_video_and_metadata(self):
        capture = _FakeCapture(fail_after=1)
        writer = None

        def writer_factory(*arguments):
            nonlocal writer
            writer = _FakeWriter(*arguments)
            return writer

        result = RtspStreamCapture(
            output_root=self.output_root,
            capture_factory=lambda: capture,
            writer_factory=writer_factory,
            cv2_module=_FakeCv2,
            now=lambda: self.fixed_now,
            monotonic=self._clock(step=0.05),
        )(
            _source(),
            1.0,
            {"MAIN_MONITOR_RTSP_URL": "rtsp://camera.local/live"},
        )

        self.assertFalse(result.captured)
        self.assertEqual(result.error_code, "stream_interrupted")
        self.assertTrue(capture.released)
        self.assertTrue(writer.released)
        self.assertEqual(
            [path for path in self.output_root.rglob("*") if path.is_file()],
            [],
        )

    def test_duration_is_bounded_before_connection(self):
        capture = _FakeCapture()
        recorder = RtspStreamCapture(
            output_root=self.output_root,
            capture_factory=lambda: capture,
            writer_factory=_FakeWriter,
            cv2_module=_FakeCv2,
            now=lambda: self.fixed_now,
            monotonic=self._clock(),
        )

        for duration in (0, 3601):
            result = recorder(
                _source(),
                duration,
                {"MAIN_MONITOR_RTSP_URL": "rtsp://camera.local/live"},
            )
            self.assertFalse(result.captured)
            self.assertEqual(result.error_code, "invalid_duration")
        self.assertIsNone(capture.open_call)


if __name__ == "__main__":
    unittest.main()
