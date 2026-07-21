from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from project_config import OUTPUTS_DIR, PROJECT_ROOT
from storage import (
    SQLiteHistoryStore,
    StreamArchiveSegmentRecord,
    StreamArchiveStateRecord,
)


ArchiveCaptureRunner = Callable[[str, Dict[str, Any]], Dict[str, Any]]


def _aware_datetime(value: str | datetime, label: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{label} 必须是 ISO 8601 时间") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} 必须包含时区")
    return parsed


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ArchiveRangeResult:
    source_id: str
    start_time: str
    end_time: str
    segments: tuple[StreamArchiveSegmentRecord, ...]
    covered_ranges: tuple[Dict[str, str], ...]
    gaps: tuple[Dict[str, str], ...]
    missing_segments: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.gaps and not self.missing_segments

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "requested_range": {
                "start_time": self.start_time,
                "end_time": self.end_time,
            },
            "complete": self.complete,
            "covered_ranges": list(self.covered_ranges),
            "gaps": list(self.gaps),
            "missing_segments": list(self.missing_segments),
            "segments": [segment.to_dict() for segment in self.segments],
        }


class HistoricalStreamArchiveManager:
    """Continuously record RTSP windows without invoking object detection."""

    def __init__(
        self,
        store: SQLiteHistoryStore,
        capture_runner: ArchiveCaptureRunner,
        *,
        output_root: Path | str = OUTPUTS_DIR / "rtsp_archive",
        allowed_recording_roots: Optional[Sequence[Path | str]] = None,
        now: Optional[Callable[[], datetime]] = None,
        retry_seconds: float = 5.0,
        recover_orphans: bool = True,
    ) -> None:
        self.store = store
        self._capture_runner = capture_runner
        self.output_root = Path(output_root).resolve()
        roots = allowed_recording_roots or (OUTPUTS_DIR / "rtsp_captures",)
        self._allowed_recording_roots = tuple(Path(root).resolve() for root in roots)
        self._now = now or (lambda: datetime.now().astimezone())
        self._retry_seconds = max(0.01, float(retry_seconds))
        self._lock = threading.Lock()
        self._stop_events: Dict[str, threading.Event] = {}
        self._threads: Dict[str, threading.Thread] = {}
        if recover_orphans:
            self.store.recover_active_stream_archives()

    def start(
        self,
        source_id: str,
        *,
        segment_seconds: float = 60.0,
        retention_seconds: float = 86400.0,
    ) -> StreamArchiveStateRecord:
        segment_seconds = float(segment_seconds)
        retention_seconds = float(retention_seconds)
        if not 1 <= segment_seconds <= 3600:
            raise ValueError("segment_seconds 必须位于 1 到 3600 之间")
        if not 3600 <= retention_seconds <= 30 * 86400:
            raise ValueError("retention_seconds 必须位于 1 小时到 30 天之间")
        with self._lock:
            thread = self._threads.get(source_id)
            if thread is not None and thread.is_alive():
                state = self.store.get_stream_archive_state(source_id)
                if state is None:
                    raise RuntimeError("录像归档线程存在但状态记录缺失")
                return state
            state = self.store.upsert_stream_archive_state(
                source_id,
                status="starting",
                segment_seconds=segment_seconds,
                retention_seconds=retention_seconds,
            )
            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._run,
                args=(source_id, stop_event),
                name=f"stream-archive-{source_id}",
                daemon=True,
            )
            self._stop_events[source_id] = stop_event
            self._threads[source_id] = thread
            thread.start()
        return state

    def stop(self, source_id: str) -> StreamArchiveStateRecord:
        state = self.store.get_stream_archive_state(source_id)
        if state is None:
            raise LookupError(f"找不到录像归档状态：{source_id}")
        if state.status in {"stopped", "failed"}:
            return state
        with self._lock:
            stop_event = self._stop_events.get(source_id)
        if stop_event is None:
            return self.store.update_stream_archive_state(
                source_id,
                status="failed",
                last_error="录像归档执行器不可用。",
            )
        state = self.store.update_stream_archive_state(
            source_id,
            status="stopping",
        )
        stop_event.set()
        return state

    def wait_for_terminal(
        self, source_id: str, *, timeout_seconds: float
    ) -> StreamArchiveStateRecord:
        with self._lock:
            thread = self._threads.get(source_id)
        if thread is not None:
            thread.join(max(0.0, timeout_seconds))
        state = self.store.get_stream_archive_state(source_id)
        if state is None:
            raise LookupError(f"找不到录像归档状态：{source_id}")
        return state

    def resolve_range(
        self,
        source_id: str,
        *,
        start_time: str | datetime,
        end_time: str | datetime,
        tolerance_seconds: float = 2.0,
    ) -> ArchiveRangeResult:
        start = _aware_datetime(start_time, "start_time").astimezone(timezone.utc)
        end = _aware_datetime(end_time, "end_time").astimezone(timezone.utc)
        if end <= start:
            raise ValueError("end_time 必须晚于 start_time")
        tolerance = timedelta(seconds=max(0.0, min(10.0, float(tolerance_seconds))))
        candidates = self.store.list_stream_archive_segments(
            source_id,
            start_time=start,
            end_time=end,
            statuses=("ready",),
        )
        usable: list[StreamArchiveSegmentRecord] = []
        missing: list[str] = []
        for segment in candidates:
            path = self._absolute_path(segment.video_path)
            if not path.is_file():
                missing.append(segment.segment_id)
            else:
                usable.append(segment)

        covered: list[Dict[str, str]] = []
        gaps: list[Dict[str, str]] = []
        cursor = start
        for segment in usable:
            segment_start = _aware_datetime(segment.started_at, "started_at").astimezone(
                timezone.utc
            )
            segment_end = _aware_datetime(segment.ended_at, "ended_at").astimezone(
                timezone.utc
            )
            overlap_start = max(start, segment_start)
            overlap_end = min(end, segment_end)
            if overlap_end <= overlap_start:
                continue
            if overlap_start > cursor + tolerance:
                gaps.append(
                    {"start_time": _utc_text(cursor), "end_time": _utc_text(overlap_start)}
                )
            cursor = max(cursor, overlap_end)
            covered.append(
                {
                    "start_time": _utc_text(overlap_start),
                    "end_time": _utc_text(overlap_end),
                }
            )
        if not usable:
            gaps.append({"start_time": _utc_text(start), "end_time": _utc_text(end)})
        elif cursor < end - tolerance:
            gaps.append({"start_time": _utc_text(cursor), "end_time": _utc_text(end)})
        return ArchiveRangeResult(
            source_id=source_id,
            start_time=_utc_text(start),
            end_time=_utc_text(end),
            segments=tuple(usable),
            covered_ranges=tuple(covered),
            gaps=tuple(gaps),
            missing_segments=tuple(missing),
        )

    def cleanup_expired(self, source_id: str) -> Dict[str, Any]:
        state = self.store.get_stream_archive_state(source_id)
        if state is None:
            raise LookupError(f"找不到录像归档状态：{source_id}")
        cutoff = self._aware_now() - timedelta(seconds=state.retention_seconds)
        expired = self.store.list_stream_archive_segments(
            source_id,
            end_time=cutoff,
            statuses=("ready", "failed"),
        )
        expired = [
            segment
            for segment in expired
            if _aware_datetime(segment.ended_at, "ended_at") <= cutoff
        ]
        deleted: list[str] = []
        refused: list[str] = []
        for segment in expired:
            paths = [segment.video_path, str(segment.metadata.get("metadata_path") or "")]
            safe = True
            for raw_path in paths:
                if not raw_path:
                    continue
                path = self._absolute_path(raw_path)
                if not self._is_allowed_recording_path(path):
                    safe = False
                    refused.append(segment.segment_id)
                    break
                if path.is_file():
                    path.unlink()
            if safe:
                self.store.mark_stream_archive_segment_deleted(segment.segment_id)
                deleted.append(segment.segment_id)
        self._write_manifest(source_id)
        return {"deleted_segment_ids": deleted, "refused_segment_ids": refused}

    def manifest_path(self, source_id: str) -> Path:
        return self.output_root / source_id / "manifest.json"

    def _run(self, source_id: str, stop_event: threading.Event) -> None:
        try:
            self.store.update_stream_archive_state(source_id, status="running", last_error="")
            while not stop_event.is_set():
                state = self.store.get_stream_archive_state(source_id)
                if state is None:
                    return
                attempt_started = self._aware_now()
                result = self._capture_runner(
                    "archive",
                    {
                        "source_id": source_id,
                        "duration_seconds": state.segment_seconds,
                    },
                )
                data = result.get("data") if isinstance(result, Mapping) else {}
                if not isinstance(data, Mapping):
                    data = {}
                if result.get("ok") and data.get("started_at") and data.get("ended_at"):
                    metadata = {
                        key: data.get(key)
                        for key in (
                            "frame_count",
                            "width",
                            "height",
                            "fps",
                            "source_codec",
                            "output_codec",
                            "backend",
                            "transport",
                            "metadata_path",
                        )
                        if data.get(key) not in (None, "")
                    }
                    segment = self.store.record_stream_archive_segment(
                        source_id,
                        started_at=str(data["started_at"]),
                        ended_at=str(data["ended_at"]),
                        status="ready",
                        video_path=str(data.get("video_path") or ""),
                        duration_seconds=float(data.get("duration_seconds") or 0.0),
                        metadata=metadata,
                    )
                    self.store.update_stream_archive_state(
                        source_id,
                        status="running",
                        last_segment_at=segment.ended_at,
                        last_error="",
                    )
                    self.cleanup_expired(source_id)
                else:
                    attempt_ended = self._aware_now()
                    if attempt_ended <= attempt_started:
                        attempt_ended = attempt_started + timedelta(microseconds=1)
                    error_message = str(
                        result.get("reply") if isinstance(result, Mapping) else "录像采集失败。"
                    )[:1000]
                    self.store.record_stream_archive_segment(
                        source_id,
                        started_at=attempt_started,
                        ended_at=attempt_ended,
                        status="failed",
                        metadata={
                            "error_code": str(
                                result.get("error_code") if isinstance(result, Mapping) else "capture_failed"
                            ),
                            "error_message": error_message,
                        },
                    )
                    self.store.update_stream_archive_state(
                        source_id,
                        status="running",
                        last_error=error_message,
                    )
                    self.cleanup_expired(source_id)
                    if stop_event.wait(self._retry_seconds):
                        break
            self.store.update_stream_archive_state(source_id, status="stopped")
        except Exception:
            try:
                self.store.update_stream_archive_state(
                    source_id,
                    status="failed",
                    last_error="录像归档执行器异常终止。",
                )
            except (LookupError, ValueError):
                pass
        finally:
            with self._lock:
                self._stop_events.pop(source_id, None)
                self._threads.pop(source_id, None)

    def _write_manifest(self, source_id: str) -> None:
        state = self.store.get_stream_archive_state(source_id)
        if state is None:
            return
        segments = self.store.list_stream_archive_segments(
            source_id,
            statuses=("ready", "failed"),
        )
        payload = {
            "source_id": source_id,
            "generated_at": self._aware_now().isoformat(timespec="seconds"),
            "segment_seconds": state.segment_seconds,
            "retention_seconds": state.retention_seconds,
            "segments": [segment.to_dict() for segment in segments],
        }
        path = self.manifest_path(source_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(path)

    def _absolute_path(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    def _is_allowed_recording_path(self, path: Path) -> bool:
        resolved = path.resolve()
        return any(resolved.is_relative_to(root) for root in self._allowed_recording_roots)

    def _aware_now(self) -> datetime:
        current = self._now()
        return current if current.tzinfo is not None else current.astimezone()
