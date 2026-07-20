from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from project_config import PROJECT_ROOT, VIDEO_SOURCES_PATH


SOURCE_SCHEMA_VERSION = 2
SUPPORTED_SOURCE_SCHEMA_VERSIONS = frozenset({1, SOURCE_SCHEMA_VERSION})
DEFAULT_SOURCE_TIMEZONE = "Asia/Shanghai"
SOURCE_KINDS = frozenset({"file", "rtsp"})
RTSP_TRANSPORTS = frozenset({"tcp", "udp", "auto"})

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
_ENV_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")


def _closed_mapping(
    raw: Any,
    *,
    allowed: set[str],
    label: str,
) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} 必须是对象")
    values = dict(raw)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"{label} 包含未知字段：{', '.join(unknown)}")
    return values


def _required_text(value: Any, label: str, *, identifier: bool = False) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} 不能为空")
    if identifier and not _ID_PATTERN.fullmatch(text):
        raise ValueError(f"{label} 只能包含小写字母、数字、下划线和连字符")
    return text


def _optional_path(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if text and Path(text).is_absolute():
        raise ValueError(f"{label} 必须使用相对于项目根目录的路径")
    return text.replace("\\", "/")


def _optional_datetime(value: Any, label: str, timezone_name: str) -> Optional[str]:
    if value in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} 必须是 ISO 8601 时间") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(timezone_name))
    return parsed.astimezone(ZoneInfo(timezone_name)).isoformat(timespec="seconds")


def _optional_positive_number(value: Any, label: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} 必须是数字或 null")
    number = float(value)
    if number <= 0:
        raise ValueError(f"{label} 必须大于 0")
    return number


def _bounded_number(
    value: Any,
    label: str,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw_value = default if value is None else value
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ValueError(f"{label} 必须是数字")
    number = float(raw_value)
    if not minimum <= number <= maximum:
        raise ValueError(f"{label} 必须在 {minimum:g} 到 {maximum:g} 之间")
    return number


def _bounded_integer(
    value: Any,
    label: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw_value = default if value is None else value
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise ValueError(f"{label} 必须是整数")
    if not minimum <= raw_value <= maximum:
        raise ValueError(f"{label} 必须在 {minimum} 到 {maximum} 之间")
    return raw_value


def _validate_rtsp_url(value: str, label: str) -> str:
    url = value.strip()
    if not url or any(character in url for character in ("\r", "\n", "\t")):
        raise ValueError(f"{label} 不是有效的 RTSP 地址")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{label} 不是有效的 RTSP 地址") from exc
    if parsed.scheme.lower() not in {"rtsp", "rtsps"} or not parsed.hostname:
        raise ValueError(f"{label} 不是有效的 RTSP 地址")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError(f"{label} 端口必须在 1 到 65535 之间")
    if parsed.fragment:
        raise ValueError(f"{label} 不能包含 URL fragment")
    return url


@dataclass(frozen=True)
class VideoResolution:
    width: int
    height: int

    @classmethod
    def from_dict(cls, raw: Any) -> "VideoResolution":
        values = _closed_mapping(
            raw,
            allowed={"width", "height"},
            label="resolution",
        )
        width = values.get("width")
        height = values.get("height")
        if (
            isinstance(width, bool)
            or isinstance(height, bool)
            or not isinstance(width, int)
            or not isinstance(height, int)
            or width <= 0
            or height <= 0
        ):
            raise ValueError("resolution.width 和 resolution.height 必须是正整数")
        return cls(width=width, height=height)

    def to_dict(self) -> Dict[str, int]:
        return {"width": self.width, "height": self.height}


@dataclass(frozen=True)
class VideoZone:
    zone_id: str
    display_name: str
    roi: tuple[int, int, int, int]

    @classmethod
    def from_dict(
        cls,
        raw: Any,
        *,
        resolution: Optional[VideoResolution] = None,
    ) -> "VideoZone":
        values = _closed_mapping(
            raw,
            allowed={"zone_id", "display_name", "roi"},
            label="zone",
        )
        roi = values.get("roi")
        if (
            not isinstance(roi, (list, tuple))
            or len(roi) != 4
            or any(isinstance(item, bool) or not isinstance(item, int) for item in roi)
        ):
            raise ValueError("zone.roi 必须是四个整数 [x1, y1, x2, y2]")
        x1, y1, x2, y2 = roi
        if min(x1, y1) < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("zone.roi 必须非负且满足 x2>x1、y2>y1")
        if resolution and (x2 > resolution.width or y2 > resolution.height):
            raise ValueError("zone.roi 不能超出视频分辨率")
        return cls(
            zone_id=_required_text(values.get("zone_id"), "zone.zone_id", identifier=True),
            display_name=_required_text(values.get("display_name"), "zone.display_name"),
            roi=(x1, y1, x2, y2),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "display_name": self.display_name,
            "roi": list(self.roi),
        }


@dataclass(frozen=True)
class VideoSegment:
    segment_id: str
    video_path: str
    started_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    start_offset_seconds: Optional[float] = None
    end_offset_seconds: Optional[float] = None

    @classmethod
    def from_dict(cls, raw: Any, *, timezone_name: str) -> "VideoSegment":
        values = _closed_mapping(
            raw,
            allowed={
                "segment_id",
                "video_path",
                "started_at",
                "duration_seconds",
                "start_offset_seconds",
                "end_offset_seconds",
            },
            label="segment",
        )
        start_offset = values.get("start_offset_seconds")
        end_offset = values.get("end_offset_seconds")
        for label, value in (
            ("segment.start_offset_seconds", start_offset),
            ("segment.end_offset_seconds", end_offset),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or float(value) < 0
            ):
                raise ValueError(f"{label} 必须是非负数字或 null")
        if (
            start_offset is not None
            and end_offset is not None
            and float(end_offset) <= float(start_offset)
        ):
            raise ValueError("segment.end_offset_seconds 必须大于 start_offset_seconds")
        return cls(
            segment_id=_required_text(
                values.get("segment_id"),
                "segment.segment_id",
                identifier=True,
            ),
            video_path=_required_text(
                _optional_path(values.get("video_path"), "segment.video_path"),
                "segment.video_path",
            ),
            started_at=_optional_datetime(
                values.get("started_at"),
                "segment.started_at",
                timezone_name,
            ),
            duration_seconds=_optional_positive_number(
                values.get("duration_seconds"),
                "segment.duration_seconds",
            ),
            start_offset_seconds=(float(start_offset) if start_offset is not None else None),
            end_offset_seconds=(float(end_offset) if end_offset is not None else None),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "video_path": self.video_path,
            "started_at": self.started_at,
            "duration_seconds": self.duration_seconds,
            "start_offset_seconds": self.start_offset_seconds,
            "end_offset_seconds": self.end_offset_seconds,
        }


@dataclass(frozen=True)
class RtspStreamSettings:
    url_env: str
    transport: str = "tcp"
    capture_window_seconds: int = 60
    segment_seconds: int = 60
    reconnect_seconds: float = 5.0
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 15.0

    @classmethod
    def from_dict(cls, raw: Any) -> "RtspStreamSettings":
        values = _closed_mapping(
            raw,
            allowed={
                "url_env",
                "transport",
                "capture_window_seconds",
                "segment_seconds",
                "reconnect_seconds",
                "connect_timeout_seconds",
                "read_timeout_seconds",
            },
            label="stream",
        )
        url_env = _required_text(values.get("url_env"), "stream.url_env")
        if not _ENV_NAME_PATTERN.fullmatch(url_env):
            raise ValueError("stream.url_env 必须是大写环境变量名")
        transport = str(values.get("transport") or "tcp").strip().lower()
        if transport not in RTSP_TRANSPORTS:
            raise ValueError("stream.transport 必须是 tcp、udp 或 auto")
        return cls(
            url_env=url_env,
            transport=transport,
            capture_window_seconds=_bounded_integer(
                values.get("capture_window_seconds"),
                "stream.capture_window_seconds",
                default=60,
                minimum=1,
                maximum=3600,
            ),
            segment_seconds=_bounded_integer(
                values.get("segment_seconds"),
                "stream.segment_seconds",
                default=60,
                minimum=5,
                maximum=3600,
            ),
            reconnect_seconds=_bounded_number(
                values.get("reconnect_seconds"),
                "stream.reconnect_seconds",
                default=5.0,
                minimum=0.1,
                maximum=300.0,
            ),
            connect_timeout_seconds=_bounded_number(
                values.get("connect_timeout_seconds"),
                "stream.connect_timeout_seconds",
                default=10.0,
                minimum=1.0,
                maximum=120.0,
            ),
            read_timeout_seconds=_bounded_number(
                values.get("read_timeout_seconds"),
                "stream.read_timeout_seconds",
                default=15.0,
                minimum=1.0,
                maximum=300.0,
            ),
        )

    def resolve_url(self, environment: Optional[Mapping[str, str]] = None) -> str:
        values = os.environ if environment is None else environment
        raw_url = str(values.get(self.url_env) or "").strip()
        if not raw_url:
            raise LookupError(f"未配置环境变量 {self.url_env}")
        return _validate_rtsp_url(raw_url, self.url_env)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url_env": self.url_env,
            "transport": self.transport,
            "capture_window_seconds": self.capture_window_seconds,
            "segment_seconds": self.segment_seconds,
            "reconnect_seconds": self.reconnect_seconds,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "read_timeout_seconds": self.read_timeout_seconds,
        }


@dataclass(frozen=True)
class LongVideoSource:
    source_id: str
    display_name: str
    source_kind: str
    video_path: str
    started_at: Optional[str]
    line_id: str
    zones: tuple[VideoZone, ...] = ()
    manifest_path: str = ""
    resolution: Optional[VideoResolution] = None
    duration_seconds: Optional[float] = None
    segments: tuple[VideoSegment, ...] = ()
    stream: Optional[RtspStreamSettings] = None

    @classmethod
    def from_dict(
        cls,
        raw: Any,
        *,
        timezone_name: str,
        schema_version: int = SOURCE_SCHEMA_VERSION,
    ) -> "LongVideoSource":
        values = _closed_mapping(
            raw,
            allowed={
                "source_id",
                "display_name",
                "source_kind",
                "video_path",
                "started_at",
                "line_id",
                "zones",
                "manifest_path",
                "resolution",
                "duration_seconds",
                "segments",
                "stream",
            },
            label="video source",
        )
        if schema_version >= 2 and "source_kind" not in values:
            raise ValueError("schema v2 的 video source 必须提供 source_kind")
        source_kind = str(values.get("source_kind") or "file").strip().lower()
        if source_kind not in SOURCE_KINDS:
            raise ValueError("source_kind 必须是 file 或 rtsp")

        resolution = (
            VideoResolution.from_dict(values["resolution"])
            if values.get("resolution") is not None
            else None
        )
        raw_zones = values.get("zones") or []
        raw_segments = values.get("segments") or []
        if not isinstance(raw_zones, list) or not isinstance(raw_segments, list):
            raise ValueError("zones 和 segments 必须是数组")
        zones = tuple(
            VideoZone.from_dict(item, resolution=resolution) for item in raw_zones
        )
        segments = tuple(
            VideoSegment.from_dict(item, timezone_name=timezone_name)
            for item in raw_segments
        )
        for label, items in (("zone_id", zones), ("segment_id", segments)):
            identifiers = [getattr(item, label) for item in items]
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"同一视频源内的 {label} 不能重复")

        stream = (
            RtspStreamSettings.from_dict(values["stream"])
            if values.get("stream") is not None
            else None
        )
        video_path = _optional_path(values.get("video_path"), "video_path")
        duration_seconds = _optional_positive_number(
            values.get("duration_seconds"),
            "duration_seconds",
        )
        if source_kind == "file" and stream is not None:
            raise ValueError("file 视频源不能配置 stream")
        if source_kind == "rtsp":
            if stream is None:
                raise ValueError("rtsp 视频源必须配置 stream")
            if video_path:
                raise ValueError("rtsp 视频源的 video_path 必须为空")
            if duration_seconds is not None:
                raise ValueError("rtsp 视频源的 duration_seconds 必须为 null")

        return cls(
            source_id=_required_text(values.get("source_id"), "source_id", identifier=True),
            display_name=_required_text(values.get("display_name"), "display_name"),
            source_kind=source_kind,
            video_path=video_path,
            started_at=_optional_datetime(
                values.get("started_at"),
                "started_at",
                timezone_name,
            ),
            line_id=_required_text(values.get("line_id"), "line_id", identifier=True),
            zones=zones,
            manifest_path=_optional_path(values.get("manifest_path"), "manifest_path"),
            resolution=resolution,
            duration_seconds=duration_seconds,
            segments=segments,
            stream=stream,
        )

    @property
    def is_file(self) -> bool:
        return self.source_kind == "file"

    @property
    def is_rtsp(self) -> bool:
        return self.source_kind == "rtsp"

    @property
    def has_archive(self) -> bool:
        return bool(self.manifest_path or self.segments)

    @property
    def is_ready(self) -> bool:
        return not self.readiness_issues()

    def resolve_stream_url(
        self,
        environment: Optional[Mapping[str, str]] = None,
    ) -> str:
        if not self.is_rtsp or self.stream is None:
            raise TypeError(f"视频源 {self.source_id} 不是 RTSP 视频源")
        return self.stream.resolve_url(environment)

    def readiness_issues(
        self,
        *,
        project_root: Path = PROJECT_ROOT,
        check_files: bool = False,
        environment: Optional[Mapping[str, str]] = None,
    ) -> tuple[str, ...]:
        issues = []
        if self.is_file:
            if not self.video_path:
                issues.append("video_path 未配置")
            if self.started_at is None:
                issues.append("started_at 未配置")
            if self.resolution is None:
                issues.append("resolution 未配置")
            if self.duration_seconds is None:
                issues.append("duration_seconds 未配置")
            if not self.has_archive:
                issues.append("manifest_path 或 segments 未配置")
        else:
            if self.stream is None:
                issues.append("stream 未配置")
            else:
                try:
                    self.resolve_stream_url(environment)
                except LookupError:
                    issues.append(f"{self.stream.url_env} 未配置")
                except ValueError:
                    issues.append(f"{self.stream.url_env} 不是有效的 RTSP 地址")

        if check_files:
            relative_paths = []
            if self.video_path:
                relative_paths.append(("video_path", self.video_path))
            if self.manifest_path:
                relative_paths.append(("manifest_path", self.manifest_path))
            relative_paths.extend(
                (f"segment[{segment.segment_id}].video_path", segment.video_path)
                for segment in self.segments
            )
            for label, relative_path in relative_paths:
                if not (project_root / relative_path).is_file():
                    issues.append(f"{label} 文件不存在：{relative_path}")
        return tuple(issues)

    def to_dict(self, *, schema_version: int = SOURCE_SCHEMA_VERSION) -> Dict[str, Any]:
        values: Dict[str, Any] = {
            "source_id": self.source_id,
            "display_name": self.display_name,
            "video_path": self.video_path,
            "started_at": self.started_at,
            "line_id": self.line_id,
            "zones": [zone.to_dict() for zone in self.zones],
            "manifest_path": self.manifest_path,
            "resolution": self.resolution.to_dict() if self.resolution else None,
            "duration_seconds": self.duration_seconds,
            "segments": [segment.to_dict() for segment in self.segments],
        }
        if schema_version >= 2:
            values = {
                "source_id": self.source_id,
                "display_name": self.display_name,
                "source_kind": self.source_kind,
                **{key: value for key, value in values.items() if key not in {"source_id", "display_name"}},
                "stream": self.stream.to_dict() if self.stream else None,
            }
        return values


@dataclass(frozen=True)
class LongVideoSourceRegistry:
    sources: tuple[LongVideoSource, ...] = field(default_factory=tuple)
    timezone: str = DEFAULT_SOURCE_TIMEZONE
    schema_version: int = SOURCE_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, raw: Any) -> "LongVideoSourceRegistry":
        values = _closed_mapping(
            raw,
            allowed={"schema_version", "timezone", "sources"},
            label="video source registry",
        )
        schema_version = values.get("schema_version")
        if schema_version not in SUPPORTED_SOURCE_SCHEMA_VERSIONS:
            supported = ", ".join(str(value) for value in sorted(SUPPORTED_SOURCE_SCHEMA_VERSIONS))
            raise ValueError(f"schema_version 必须是受支持版本：{supported}")
        timezone_name = _required_text(values.get("timezone"), "timezone")
        try:
            ZoneInfo(timezone_name)
        except Exception as exc:
            raise ValueError(f"无效时区：{timezone_name}") from exc
        raw_sources = values.get("sources")
        if not isinstance(raw_sources, list):
            raise ValueError("sources 必须是数组")
        sources = tuple(
            LongVideoSource.from_dict(
                item,
                timezone_name=timezone_name,
                schema_version=schema_version,
            )
            for item in raw_sources
        )
        source_ids = [source.source_id for source in sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_id 不能重复")
        return cls(
            sources=sources,
            timezone=timezone_name,
            schema_version=schema_version,
        )

    @classmethod
    def load(cls, path: Path | str = VIDEO_SOURCES_PATH) -> "LongVideoSourceRegistry":
        registry_path = Path(path)
        if not registry_path.is_file():
            raise FileNotFoundError(f"找不到长视频源配置：{registry_path}")
        return cls.from_dict(json.loads(registry_path.read_text(encoding="utf-8")))

    def get(self, source_id: str) -> LongVideoSource:
        normalized = source_id.strip()
        for source in self.sources:
            if source.source_id == normalized:
                return source
        raise LookupError(f"未注册的长视频源：{source_id}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "timezone": self.timezone,
            "sources": [
                source.to_dict(schema_version=self.schema_version) for source in self.sources
            ],
        }

    def unresolved_sources(
        self,
        *,
        environment: Optional[Mapping[str, str]] = None,
    ) -> tuple[LongVideoSource, ...]:
        return tuple(
            source
            for source in self.sources
            if source.readiness_issues(environment=environment)
        )


def load_video_source_registry(
    path: Path | str = VIDEO_SOURCES_PATH,
) -> LongVideoSourceRegistry:
    return LongVideoSourceRegistry.load(path)
