from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from project_config import OUTPUTS_DIR

from .video_sources import LongVideoSource


_FFMPEG_OPTIONS_LOCK = threading.Lock()


@dataclass(frozen=True)
class StreamProbeResult:
    """Safe metadata returned by an RTSP connection probe.

    The resolved stream URL is deliberately not part of this object, so it
    cannot accidentally reach chat output, history, or logs.
    """

    source_id: str
    display_name: str
    line_id: str
    online: bool
    checked_at: str
    latency_ms: int
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    codec: str = ""
    backend: str = ""
    transport: str = ""
    error_code: str = ""
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StreamCaptureResult:
    """Safe metadata for one bounded local recording from an RTSP source."""

    source_id: str
    display_name: str
    line_id: str
    captured: bool
    requested_duration_seconds: float
    started_at: str = ""
    ended_at: str = ""
    duration_seconds: float = 0.0
    frame_count: int = 0
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    source_codec: str = ""
    output_codec: str = ""
    backend: str = ""
    transport: str = ""
    video_path: str = ""
    metadata_path: str = ""
    error_code: str = ""
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RtspStreamProbe:
    """Open an RTSP source and read exactly one frame without running YOLO."""

    def __init__(
        self,
        *,
        capture_factory: Optional[Callable[[], Any]] = None,
        cv2_module: Any = None,
        now: Optional[Callable[[], datetime]] = None,
        monotonic: Optional[Callable[[], float]] = None,
    ) -> None:
        self._capture_factory = capture_factory
        self._cv2_module = cv2_module
        self._now = now or (lambda: datetime.now().astimezone())
        self._monotonic = monotonic or time.monotonic

    def __call__(
        self,
        source: LongVideoSource,
        environment: Optional[Mapping[str, str]] = None,
    ) -> StreamProbeResult:
        checked_at = self._checked_at()
        started = self._monotonic()
        transport = source.stream.transport if source.stream else ""

        if not source.is_rtsp or source.stream is None:
            return self._failure(
                source,
                checked_at=checked_at,
                started=started,
                transport=transport,
                error_code="not_rtsp_source",
                error_message="该视频源不是 RTSP 视频源。",
            )

        try:
            url = source.resolve_stream_url(environment)
        except (LookupError, ValueError, TypeError):
            return self._failure(
                source,
                checked_at=checked_at,
                started=started,
                transport=transport,
                error_code="configuration_error",
                error_message=f"视频源连接地址未正确配置，请检查环境变量 {source.stream.url_env}。",
            )

        capture = None
        try:
            cv2 = self._load_cv2()
            capture = (
                self._capture_factory()
                if self._capture_factory is not None
                else cv2.VideoCapture()
            )
            opened = self._open_capture(capture, cv2, url, source)
            open_elapsed = self._monotonic() - started
            if not opened or not capture.isOpened():
                timed_out = open_elapsed >= source.stream.connect_timeout_seconds
                return self._failure(
                    source,
                    checked_at=checked_at,
                    started=started,
                    transport=transport,
                    error_code="connection_timeout" if timed_out else "connection_failed",
                    error_message=(
                        "连接视频源超时。" if timed_out else "无法连接视频源。"
                    ),
                )

            read_started = self._monotonic()
            frame_ok, frame = capture.read()
            read_elapsed = self._monotonic() - read_started
            if not frame_ok or frame is None:
                timed_out = read_elapsed >= source.stream.read_timeout_seconds
                return self._failure(
                    source,
                    checked_at=checked_at,
                    started=started,
                    transport=transport,
                    error_code="connection_timeout" if timed_out else "no_video_frame",
                    error_message=(
                        "读取视频帧超时。" if timed_out else "连接已建立，但未读取到视频帧。"
                    ),
                )

            width = self._positive_int(self._capture_value(capture, cv2, "CAP_PROP_FRAME_WIDTH"))
            height = self._positive_int(self._capture_value(capture, cv2, "CAP_PROP_FRAME_HEIGHT"))
            shape = getattr(frame, "shape", ())
            if (width is None or height is None) and len(shape) >= 2:
                height = height or self._positive_int(shape[0])
                width = width or self._positive_int(shape[1])
            fps = self._positive_float(self._capture_value(capture, cv2, "CAP_PROP_FPS"))
            fourcc = self._capture_value(capture, cv2, "CAP_PROP_FOURCC")
            return StreamProbeResult(
                source_id=source.source_id,
                display_name=source.display_name,
                line_id=source.line_id,
                online=True,
                checked_at=checked_at,
                latency_ms=self._latency_ms(started),
                width=width,
                height=height,
                fps=round(fps, 3) if fps is not None else None,
                codec=self._codec_name(fourcc),
                backend=self._backend_name(capture),
                transport=transport,
            )
        except Exception:
            # Native OpenCV/FFmpeg exceptions may contain the full URL. Never
            # surface their raw text across the Skill boundary.
            return self._failure(
                source,
                checked_at=checked_at,
                started=started,
                transport=transport,
                error_code="probe_failed",
                error_message="视频源探测失败。",
            )
        finally:
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    pass

    def _load_cv2(self) -> Any:
        if self._cv2_module is not None:
            return self._cv2_module
        import cv2

        return cv2

    def _open_capture(
        self,
        capture: Any,
        cv2: Any,
        url: str,
        source: LongVideoSource,
    ) -> bool:
        settings = source.stream
        if settings is None:
            return False
        parameters = []
        open_timeout = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
        read_timeout = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
        if open_timeout is not None:
            parameters.extend([open_timeout, int(settings.connect_timeout_seconds * 1000)])
        if read_timeout is not None:
            parameters.extend([read_timeout, int(settings.read_timeout_seconds * 1000)])

        backend = getattr(cv2, "CAP_FFMPEG", 0)
        if settings.transport == "auto":
            return bool(capture.open(url, backend, parameters))

        option_name = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
        option_value = f"rtsp_transport;{settings.transport}"
        with _FFMPEG_OPTIONS_LOCK:
            previous = os.environ.get(option_name)
            os.environ[option_name] = option_value
            try:
                return bool(capture.open(url, backend, parameters))
            finally:
                if previous is None:
                    os.environ.pop(option_name, None)
                else:
                    os.environ[option_name] = previous

    @staticmethod
    def _capture_value(capture: Any, cv2: Any, property_name: str) -> Any:
        property_id = getattr(cv2, property_name, None)
        if property_id is None:
            return None
        try:
            return capture.get(property_id)
        except Exception:
            return None

    @staticmethod
    def _positive_int(value: Any) -> Optional[int]:
        try:
            number = int(round(float(value)))
        except (TypeError, ValueError, OverflowError):
            return None
        return number if number > 0 else None

    @staticmethod
    def _positive_float(value: Any) -> Optional[float]:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if number > 0 else None

    @staticmethod
    def _codec_name(fourcc_value: Any) -> str:
        try:
            fourcc = int(fourcc_value)
        except (TypeError, ValueError, OverflowError):
            return ""
        if fourcc <= 0:
            return ""
        raw = "".join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4))
        normalized = raw.strip("\x00 ").lower()
        aliases = {
            "avc1": "h264",
            "h264": "h264",
            "x264": "h264",
            "hevc": "h265",
            "h265": "h265",
            "hev1": "h265",
            "hvc1": "h265",
        }
        return aliases.get(normalized, normalized)

    @staticmethod
    def _backend_name(capture: Any) -> str:
        try:
            return str(capture.getBackendName() or "")
        except Exception:
            return ""

    def _checked_at(self) -> str:
        current = self._now()
        if current.tzinfo is None:
            current = current.astimezone()
        return current.isoformat(timespec="seconds")

    def _latency_ms(self, started: float) -> int:
        return max(0, int(round((self._monotonic() - started) * 1000)))

    def _failure(
        self,
        source: LongVideoSource,
        *,
        checked_at: str,
        started: float,
        transport: str,
        error_code: str,
        error_message: str,
    ) -> StreamProbeResult:
        return StreamProbeResult(
            source_id=source.source_id,
            display_name=source.display_name,
            line_id=source.line_id,
            online=False,
            checked_at=checked_at,
            latency_ms=self._latency_ms(started),
            transport=transport,
            error_code=error_code,
            error_message=error_message,
        )


class RtspStreamCapture:
    """Record a bounded RTSP window to MP4 without invoking object detection."""

    def __init__(
        self,
        *,
        output_root: Path | str = OUTPUTS_DIR / "rtsp_captures",
        capture_factory: Optional[Callable[[], Any]] = None,
        writer_factory: Optional[Callable[..., Any]] = None,
        cv2_module: Any = None,
        now: Optional[Callable[[], datetime]] = None,
        monotonic: Optional[Callable[[], float]] = None,
    ) -> None:
        self._output_root = Path(output_root)
        self._capture_factory = capture_factory
        self._writer_factory = writer_factory
        self._cv2_module = cv2_module
        self._now = now or (lambda: datetime.now().astimezone())
        self._monotonic = monotonic or time.monotonic

    def __call__(
        self,
        source: LongVideoSource,
        duration_seconds: Optional[float] = None,
        environment: Optional[Mapping[str, str]] = None,
    ) -> StreamCaptureResult:
        settings = source.stream
        requested_duration = float(
            duration_seconds
            if duration_seconds is not None
            else (settings.capture_window_seconds if settings else 0)
        )
        transport = settings.transport if settings else ""
        if not 1 <= requested_duration <= 3600:
            return self._failure(
                source,
                requested_duration=requested_duration,
                transport=transport,
                error_code="invalid_duration",
                error_message="采集时长必须在 1 到 3600 秒之间。",
            )
        if not source.is_rtsp or settings is None:
            return self._failure(
                source,
                requested_duration=requested_duration,
                transport=transport,
                error_code="not_rtsp_source",
                error_message="该视频源不是 RTSP 视频源。",
            )
        try:
            url = source.resolve_stream_url(environment)
        except (LookupError, ValueError, TypeError):
            return self._failure(
                source,
                requested_duration=requested_duration,
                transport=transport,
                error_code="configuration_error",
                error_message=f"视频源连接地址未正确配置，请检查环境变量 {settings.url_env}。",
            )

        capture = None
        writer = None
        temp_video_path: Optional[Path] = None
        final_video_path: Optional[Path] = None
        temp_metadata_path: Optional[Path] = None
        metadata_path: Optional[Path] = None
        completed = False
        try:
            cv2 = self._load_cv2()
            capture = (
                self._capture_factory()
                if self._capture_factory is not None
                else cv2.VideoCapture()
            )
            adapter = RtspStreamProbe(cv2_module=cv2)
            connection_started = self._monotonic()
            opened = adapter._open_capture(capture, cv2, url, source)
            connection_elapsed = self._monotonic() - connection_started
            if not opened or not capture.isOpened():
                timed_out = connection_elapsed >= settings.connect_timeout_seconds
                return self._failure(
                    source,
                    requested_duration=requested_duration,
                    transport=transport,
                    error_code="connection_timeout" if timed_out else "connection_failed",
                    error_message="连接视频源超时。" if timed_out else "无法连接视频源。",
                )

            read_started = self._monotonic()
            frame_ok, frame = capture.read()
            read_elapsed = self._monotonic() - read_started
            if not frame_ok or frame is None:
                timed_out = read_elapsed >= settings.read_timeout_seconds
                return self._failure(
                    source,
                    requested_duration=requested_duration,
                    transport=transport,
                    error_code="connection_timeout" if timed_out else "no_video_frame",
                    error_message=(
                        "读取视频帧超时。" if timed_out else "连接已建立，但未读取到视频帧。"
                    ),
                )

            width = adapter._positive_int(
                adapter._capture_value(capture, cv2, "CAP_PROP_FRAME_WIDTH")
            )
            height = adapter._positive_int(
                adapter._capture_value(capture, cv2, "CAP_PROP_FRAME_HEIGHT")
            )
            shape = getattr(frame, "shape", ())
            if (width is None or height is None) and len(shape) >= 2:
                height = height or adapter._positive_int(shape[0])
                width = width or adapter._positive_int(shape[1])
            if width is None or height is None:
                return self._failure(
                    source,
                    requested_duration=requested_duration,
                    transport=transport,
                    error_code="invalid_stream_metadata",
                    error_message="视频流缺少有效的画面尺寸。",
                )
            reported_fps = adapter._positive_float(
                adapter._capture_value(capture, cv2, "CAP_PROP_FPS")
            )
            fps = (
                reported_fps
                if reported_fps is not None and reported_fps <= 120
                else 25.0
            )
            source_fourcc = adapter._capture_value(capture, cv2, "CAP_PROP_FOURCC")

            captured_at = self._aware_now()
            capture_id = (
                f"{source.source_id}_{captured_at.strftime('%Y%m%d_%H%M%S_%f')}"
            )
            capture_dir = self._output_root / source.source_id
            capture_dir.mkdir(parents=True, exist_ok=True)
            final_video_path = capture_dir / f"{capture_id}.mp4"
            temp_video_path = capture_dir / f".{capture_id}.part.mp4"
            metadata_path = capture_dir / f"{capture_id}.json"
            temp_metadata_path = capture_dir / f".{capture_id}.part.json"

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = (
                self._writer_factory(str(temp_video_path), fourcc, fps, (width, height))
                if self._writer_factory is not None
                else cv2.VideoWriter(
                    str(temp_video_path),
                    fourcc,
                    fps,
                    (width, height),
                )
            )
            if not writer.isOpened():
                return self._failure(
                    source,
                    requested_duration=requested_duration,
                    transport=transport,
                    error_code="writer_failed",
                    error_message="无法创建本地 MP4 采集文件。",
                )

            capture_started = self._monotonic()
            target_frame_count = max(1, int(round(requested_duration * fps)))
            frame_count = 1
            writer.write(frame)
            while True:
                elapsed = self._monotonic() - capture_started
                if elapsed >= requested_duration:
                    break
                frame_ok, frame = capture.read()
                if not frame_ok or frame is None:
                    return self._failure(
                        source,
                        requested_duration=requested_duration,
                        transport=transport,
                        error_code="stream_interrupted",
                        error_message="视频流在采集完成前中断。",
                    )
                elapsed = self._monotonic() - capture_started
                expected_frame_count = min(
                    target_frame_count,
                    max(1, int(elapsed * fps) + 1),
                )
                while frame_count < expected_frame_count:
                    writer.write(frame)
                    frame_count += 1

            # If a source delivers frames slightly slower than its declared
            # FPS, duplicate the latest valid frame so the encoded duration
            # still matches the requested wall-clock window.
            while frame_count < target_frame_count:
                writer.write(frame)
                frame_count += 1

            writer.release()
            writer = None
            if not temp_video_path.is_file() or temp_video_path.stat().st_size <= 0:
                return self._failure(
                    source,
                    requested_duration=requested_duration,
                    transport=transport,
                    error_code="empty_capture",
                    error_message="采集完成但没有生成有效视频文件。",
                )
            temp_video_path.replace(final_video_path)
            encoded_duration = round(frame_count / fps, 3)
            result = StreamCaptureResult(
                source_id=source.source_id,
                display_name=source.display_name,
                line_id=source.line_id,
                captured=True,
                requested_duration_seconds=requested_duration,
                started_at=captured_at.isoformat(timespec="seconds"),
                ended_at=self._aware_now().isoformat(timespec="seconds"),
                duration_seconds=encoded_duration,
                frame_count=frame_count,
                width=width,
                height=height,
                fps=round(fps, 3),
                source_codec=adapter._codec_name(source_fourcc),
                output_codec="mp4v",
                backend=adapter._backend_name(capture),
                transport=transport,
                video_path=str(final_video_path.resolve()),
                metadata_path=str(metadata_path.resolve()),
            )
            temp_metadata_path.write_text(
                json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temp_metadata_path.replace(metadata_path)
            completed = True
            return result
        except Exception:
            # Do not expose OpenCV/FFmpeg exception text because it can contain
            # the resolved RTSP URL and credentials.
            return self._failure(
                source,
                requested_duration=requested_duration,
                transport=transport,
                error_code="capture_failed",
                error_message="视频流采集失败。",
            )
        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    pass
            if not completed:
                for path in (
                    temp_video_path,
                    final_video_path,
                    temp_metadata_path,
                    metadata_path,
                ):
                    if path is not None:
                        try:
                            path.unlink(missing_ok=True)
                        except OSError:
                            pass

    def _load_cv2(self) -> Any:
        if self._cv2_module is not None:
            return self._cv2_module
        import cv2

        return cv2

    def _aware_now(self) -> datetime:
        current = self._now()
        return current if current.tzinfo is not None else current.astimezone()

    @staticmethod
    def _failure(
        source: LongVideoSource,
        *,
        requested_duration: float,
        transport: str,
        error_code: str,
        error_message: str,
    ) -> StreamCaptureResult:
        return StreamCaptureResult(
            source_id=source.source_id,
            display_name=source.display_name,
            line_id=source.line_id,
            captured=False,
            requested_duration_seconds=requested_duration,
            transport=transport,
            error_code=error_code,
            error_message=error_message,
        )
