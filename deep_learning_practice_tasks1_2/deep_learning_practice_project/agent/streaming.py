from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Mapping, Optional

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
