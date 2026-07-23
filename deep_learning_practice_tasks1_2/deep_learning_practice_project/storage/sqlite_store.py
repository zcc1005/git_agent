from __future__ import annotations

import json
import hashlib
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional


VALID_ALARM_STATUSES = {"inactive", "pending", "confirmed", "cancelled"}
VALID_ALARM_ACTIONS = {"confirm", "cancel"}
VALID_REVIEW_STATUSES = {"unreviewed", "confirmed", "rejected", "closed"}
VALID_REVIEW_ACTIONS = {"confirm", "reject", "close", "reopen"}
VALID_MONITORING_STATUSES = {
    "scheduled",
    "running",
    "stop_requested",
    "completed",
    "stopped",
    "failed",
    "interrupted",
}
VALID_MONITORING_JOB_STATUSES = {
    "pending",
    "connecting",
    "running",
    "stopping",
    "completed",
    "failed",
    "cancelled",
}
VALID_STREAM_SEGMENT_STATUSES = {"pending", "processing", "completed", "failed"}
VALID_STREAM_ARCHIVE_STATUSES = {
    "stopped",
    "starting",
    "running",
    "stopping",
    "failed",
}
VALID_STREAM_ARCHIVE_SEGMENT_STATUSES = {"recording", "ready", "failed", "deleted"}
VALID_REALTIME_INSPECTION_STATUSES = {
    "scheduled", "connecting", "running", "reconnecting", "stop_requested",
    "completed", "stopped", "failed", "interrupted",
}
ACTIVE_REALTIME_INSPECTION_STATUSES = (
    "scheduled", "connecting", "running", "reconnecting", "stop_requested",
)
MONITORING_TASK_TO_JOB_STATUS = {
    "scheduled": "pending",
    "running": "running",
    "stop_requested": "stopping",
    "completed": "completed",
    "stopped": "cancelled",
    "failed": "failed",
    "interrupted": "failed",
}


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_load(value: str) -> Any:
    return json.loads(value) if value else {}


def _normalized_utc_time(value: str | datetime, label: str) -> str:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError(f"{label} 必须是 ISO 8601 时间") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} 必须包含时区")
    # Preserve sub-second precision for short bounded background tasks and tests.
    return parsed.astimezone(timezone.utc).isoformat()


@dataclass(frozen=True)
class DetectionRecord:
    id: str
    session_id: str
    source_type: str
    source_path: str
    status: str
    risk_level: str
    summary: Dict[str, Any]
    alarm_report: str
    line_id: str
    source_started_at: str
    source_ended_at: str
    review_status: str
    review_note: str
    reviewer: str
    reviewed_at: str
    created_at: str


@dataclass(frozen=True)
class AlarmRecord:
    id: str
    detection_id: str
    session_id: str
    risk_level: str
    status: str
    requires_stop: bool
    report: Dict[str, Any]
    report_text: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class MonitoringTaskRecord:
    id: str
    session_id: str
    source_id: str
    line_id: str
    zone_id: str
    status: str
    start_time: str
    end_time: str
    config: Dict[str, Any]
    runs_completed: int
    runs_succeeded: int
    runs_failed: int
    consecutive_failures: int
    last_run_started_at: str
    last_run_ended_at: str
    last_detection_id: str
    last_alarm_id: str
    last_risk_level: str
    last_error_code: str
    last_error_message: str
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.id,
            "session_id": self.session_id,
            "source_id": self.source_id,
            "line_id": self.line_id,
            "zone_id": self.zone_id,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "config": self.config,
            "runs_completed": self.runs_completed,
            "runs_succeeded": self.runs_succeeded,
            "runs_failed": self.runs_failed,
            "consecutive_failures": self.consecutive_failures,
            "last_run_started_at": self.last_run_started_at,
            "last_run_ended_at": self.last_run_ended_at,
            "last_detection_id": self.last_detection_id,
            "last_alarm_id": self.last_alarm_id,
            "last_risk_level": self.last_risk_level,
            "last_error_code": self.last_error_code,
            "last_error_message": self.last_error_message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class MonitoringRunRecord:
    id: int
    task_id: str
    run_index: int
    status: str
    started_at: str
    ended_at: str
    detection_id: str
    alarm_id: str
    risk_level: str
    error_code: str
    error_message: str
    result: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.id,
            "task_id": self.task_id,
            "run_index": self.run_index,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "detection_id": self.detection_id,
            "alarm_id": self.alarm_id,
            "risk_level": self.risk_level,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "result": self.result,
        }


@dataclass(frozen=True)
class MonitoringJobRecord:
    task_id: str
    source_id: str
    status: str
    started_at: str
    ends_at: str
    segment_seconds: float
    last_processed_at: str
    last_error: str
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "source_id": self.source_id,
            "status": self.status,
            "started_at": self.started_at,
            "ends_at": self.ends_at,
            "segment_seconds": self.segment_seconds,
            "last_processed_at": self.last_processed_at,
            "last_error": self.last_error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class StreamSegmentRecord:
    segment_id: str
    task_id: str
    source_id: str
    video_path: str
    started_at: str
    ended_at: str
    status: str
    detection_id: str
    retry_count: int
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "task_id": self.task_id,
            "source_id": self.source_id,
            "video_path": self.video_path,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "detection_id": self.detection_id,
            "retry_count": self.retry_count,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class StreamArchiveStateRecord:
    source_id: str
    status: str
    segment_seconds: float
    retention_seconds: float
    last_segment_at: str
    last_error: str
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "status": self.status,
            "segment_seconds": self.segment_seconds,
            "retention_seconds": self.retention_seconds,
            "retention_hours": self.retention_seconds / 3600.0,
            "last_segment_at": self.last_segment_at,
            "last_error": self.last_error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class StreamArchiveSegmentRecord:
    segment_id: str
    source_id: str
    video_path: str
    started_at: str
    ended_at: str
    duration_seconds: float
    status: str
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "source_id": self.source_id,
            "video_path": self.video_path,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "status": self.status,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class RealtimeInspectionTaskRecord:
    id: str
    session_id: str
    source_id: str
    line_id: str
    zone_id: str
    start_time: str
    end_time: str
    status: str
    sample_fps: float
    created_at: str
    updated_at: str
    started_at: str
    stopped_at: str
    frames_read: int
    frames_inferred: int
    frames_dropped: int
    inference_failures: int
    reconnect_count: int
    events_detected: int
    alarms_created: int
    highest_risk_level: str
    last_frame_at: str
    last_inference_at: str
    latest_detection_id: str
    latest_alarm_id: str
    latest_event_frame: str
    last_error_code: str
    last_error_message: str
    config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        values = dict(self.__dict__)
        values["task_id"] = values.pop("id")
        started = self.started_at or self.start_time
        try:
            start = datetime.fromisoformat(started.replace("Z", "+00:00"))
            finish_text = self.stopped_at or self.updated_at
            finish = datetime.fromisoformat(finish_text.replace("Z", "+00:00"))
            values["elapsed_seconds"] = round(max(0.0, (finish - start).total_seconds()), 3)
        except (TypeError, ValueError):
            values["elapsed_seconds"] = 0.0
        elapsed = float(values["elapsed_seconds"])
        values["inference_fps"] = round(self.frames_inferred / elapsed, 3) if elapsed else 0.0
        return values


@dataclass(frozen=True)
class RealtimeInspectionEventRecord:
    event_id: str
    task_id: str
    source_id: str
    detected_at: str
    ended_at: str
    class_name: str
    confidence: float
    bbox: List[float]
    risk_level: str
    detection_id: str
    alarm_id: str
    image_path: str
    metadata: Dict[str, Any]
    created_at: str
    line_id: str
    last_seen_at: str
    event_status: str
    hit_count: int
    class_counts: Dict[str, int]
    max_confidence: float
    alarm_report: str
    llm_summary: str
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        values = dict(self.__dict__)
        values["representative_frame"] = self.image_path
        values["confidence"] = self.confidence
        return values


class SQLiteHistoryStore:
    """Small SQLite repository used by the agent and future Web API.

    A connection is opened per operation so the store is safe to share between
    Flask request threads.  SQLite remains the source of truth for conversation
    history and alarm acknowledgement state; model output files stay unchanged.
    """

    def __init__(
        self,
        db_path: Path | str,
        now: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.db_path = Path(db_path)
        self._now = now or (lambda: datetime.now().astimezone())
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _timestamp(self) -> str:
        return self._now().isoformat(timespec="seconds")

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    intent TEXT NOT NULL DEFAULT '',
                    tool_name TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS detection_runs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    source_type TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    alarm_report TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS alarms (
                    id TEXT PRIMARY KEY,
                    detection_id TEXT NOT NULL UNIQUE
                        REFERENCES detection_runs(id) ON DELETE CASCADE,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    risk_level TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('inactive', 'pending', 'confirmed', 'cancelled')
                    ),
                    requires_stop INTEGER NOT NULL DEFAULT 0,
                    report_json TEXT NOT NULL,
                    report_text TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS alarm_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alarm_id TEXT NOT NULL REFERENCES alarms(id) ON DELETE CASCADE,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    action TEXT NOT NULL CHECK (action IN ('confirm', 'cancel')),
                    note TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS detection_review_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    detection_id TEXT NOT NULL
                        REFERENCES detection_runs(id) ON DELETE CASCADE,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    action TEXT NOT NULL CHECK (
                        action IN ('confirm', 'reject', 'close', 'reopen')
                    ),
                    reviewer TEXT NOT NULL DEFAULT '',
                    note TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS monitoring_tasks (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    source_id TEXT NOT NULL,
                    line_id TEXT NOT NULL DEFAULT '',
                    zone_id TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL CHECK (
                        status IN (
                            'scheduled', 'running', 'stop_requested', 'completed',
                            'stopped', 'failed', 'interrupted'
                        )
                    ),
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    config_json TEXT NOT NULL DEFAULT '{}',
                    runs_completed INTEGER NOT NULL DEFAULT 0,
                    runs_succeeded INTEGER NOT NULL DEFAULT 0,
                    runs_failed INTEGER NOT NULL DEFAULT 0,
                    consecutive_failures INTEGER NOT NULL DEFAULT 0,
                    last_run_started_at TEXT NOT NULL DEFAULT '',
                    last_run_ended_at TEXT NOT NULL DEFAULT '',
                    last_detection_id TEXT NOT NULL DEFAULT '',
                    last_alarm_id TEXT NOT NULL DEFAULT '',
                    last_risk_level TEXT NOT NULL DEFAULT '',
                    last_error_code TEXT NOT NULL DEFAULT '',
                    last_error_message TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS monitoring_task_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL REFERENCES monitoring_tasks(id) ON DELETE CASCADE,
                    run_index INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('succeeded', 'failed')),
                    started_at TEXT NOT NULL,
                    ended_at TEXT NOT NULL,
                    detection_id TEXT NOT NULL DEFAULT '',
                    alarm_id TEXT NOT NULL DEFAULT '',
                    risk_level TEXT NOT NULL DEFAULT '',
                    error_code TEXT NOT NULL DEFAULT '',
                    error_message TEXT NOT NULL DEFAULT '',
                    result_json TEXT NOT NULL DEFAULT '{}',
                    UNIQUE(task_id, run_index)
                );

                CREATE TABLE IF NOT EXISTS monitoring_jobs (
                    task_id TEXT PRIMARY KEY
                        REFERENCES monitoring_tasks(id) ON DELETE CASCADE,
                    source_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN (
                            'pending', 'connecting', 'running', 'stopping',
                            'completed', 'failed', 'cancelled'
                        )
                    ),
                    started_at TEXT NOT NULL,
                    ends_at TEXT NOT NULL,
                    segment_seconds REAL NOT NULL CHECK (segment_seconds > 0),
                    last_processed_at TEXT NOT NULL DEFAULT '',
                    last_error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS stream_segments (
                    segment_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL
                        REFERENCES monitoring_jobs(task_id) ON DELETE CASCADE,
                    source_id TEXT NOT NULL,
                    video_path TEXT NOT NULL DEFAULT '',
                    started_at TEXT NOT NULL,
                    ended_at TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('pending', 'processing', 'completed', 'failed')
                    ),
                    detection_id TEXT NOT NULL DEFAULT '',
                    retry_count INTEGER NOT NULL DEFAULT 0 CHECK (retry_count >= 0),
                    created_at TEXT NOT NULL,
                    UNIQUE(task_id, source_id, started_at, ended_at)
                );

                CREATE TABLE IF NOT EXISTS stream_archive_state (
                    source_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK (
                        status IN ('stopped', 'starting', 'running', 'stopping', 'failed')
                    ),
                    segment_seconds REAL NOT NULL CHECK (segment_seconds >= 1),
                    retention_seconds REAL NOT NULL CHECK (retention_seconds >= 3600),
                    last_segment_at TEXT NOT NULL DEFAULT '',
                    last_error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS stream_archive_segments (
                    segment_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    video_path TEXT NOT NULL DEFAULT '',
                    started_at TEXT NOT NULL,
                    ended_at TEXT NOT NULL,
                    duration_seconds REAL NOT NULL CHECK (duration_seconds >= 0),
                    status TEXT NOT NULL CHECK (
                        status IN ('recording', 'ready', 'failed', 'deleted')
                    ),
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(source_id, started_at, ended_at)
                );

                CREATE TABLE IF NOT EXISTS realtime_inspection_tasks (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    source_id TEXT NOT NULL,
                    line_id TEXT NOT NULL DEFAULT '',
                    zone_id TEXT NOT NULL DEFAULT '',
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN (
                        'scheduled', 'connecting', 'running', 'reconnecting',
                        'stop_requested', 'completed', 'stopped', 'failed', 'interrupted'
                    )),
                    sample_fps REAL NOT NULL CHECK(sample_fps >= 0.2 AND sample_fps <= 10),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT NOT NULL DEFAULT '',
                    stopped_at TEXT NOT NULL DEFAULT '',
                    frames_read INTEGER NOT NULL DEFAULT 0,
                    frames_inferred INTEGER NOT NULL DEFAULT 0,
                    frames_dropped INTEGER NOT NULL DEFAULT 0,
                    inference_failures INTEGER NOT NULL DEFAULT 0,
                    reconnect_count INTEGER NOT NULL DEFAULT 0,
                    events_detected INTEGER NOT NULL DEFAULT 0,
                    alarms_created INTEGER NOT NULL DEFAULT 0,
                    highest_risk_level TEXT NOT NULL DEFAULT 'none',
                    last_frame_at TEXT NOT NULL DEFAULT '',
                    last_inference_at TEXT NOT NULL DEFAULT '',
                    latest_detection_id TEXT NOT NULL DEFAULT '',
                    latest_alarm_id TEXT NOT NULL DEFAULT '',
                    latest_event_frame TEXT NOT NULL DEFAULT '',
                    last_error_code TEXT NOT NULL DEFAULT '',
                    last_error_message TEXT NOT NULL DEFAULT '',
                    config_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS realtime_inspection_events (
                    event_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES realtime_inspection_tasks(id) ON DELETE CASCADE,
                    source_id TEXT NOT NULL,
                    detected_at TEXT NOT NULL,
                    ended_at TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    bbox_json TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    detection_id TEXT NOT NULL DEFAULT '',
                    alarm_id TEXT NOT NULL DEFAULT '',
                    image_path TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    line_id TEXT NOT NULL DEFAULT '',
                    last_seen_at TEXT NOT NULL DEFAULT '',
                    event_status TEXT NOT NULL DEFAULT 'closed',
                    hit_count INTEGER NOT NULL DEFAULT 0,
                    class_counts_json TEXT NOT NULL DEFAULT '{}',
                    max_confidence REAL NOT NULL DEFAULT 0,
                    alarm_report TEXT NOT NULL DEFAULT '',
                    llm_summary TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT '',
                    UNIQUE(task_id, event_id)
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session_created
                    ON messages(session_id, created_at, id);
                CREATE INDEX IF NOT EXISTS idx_detection_session_created
                    ON detection_runs(session_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_alarm_session_created
                    ON alarms(session_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_alarm_level_created
                    ON alarms(risk_level, created_at);
                CREATE INDEX IF NOT EXISTS idx_monitoring_session_created
                    ON monitoring_tasks(session_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_monitoring_source_status
                    ON monitoring_tasks(source_id, status, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_monitoring_runs_task
                    ON monitoring_task_runs(task_id, run_index DESC);
                CREATE INDEX IF NOT EXISTS idx_monitoring_jobs_source_status
                    ON monitoring_jobs(source_id, status, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_stream_segments_task_started
                    ON stream_segments(task_id, started_at DESC);
                CREATE INDEX IF NOT EXISTS idx_stream_segments_detection
                    ON stream_segments(detection_id);
                CREATE UNIQUE INDEX IF NOT EXISTS uq_stream_segments_video_path
                    ON stream_segments(video_path)
                    WHERE video_path <> '';
                CREATE INDEX IF NOT EXISTS idx_archive_segments_source_time
                    ON stream_archive_segments(source_id, started_at, ended_at);
                CREATE INDEX IF NOT EXISTS idx_archive_segments_source_status
                    ON stream_archive_segments(source_id, status, ended_at DESC);
                CREATE UNIQUE INDEX IF NOT EXISTS uq_archive_segments_video_path
                    ON stream_archive_segments(video_path)
                    WHERE video_path <> '';
                CREATE INDEX IF NOT EXISTS idx_realtime_session_created
                    ON realtime_inspection_tasks(session_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_realtime_source_status
                    ON realtime_inspection_tasks(source_id, status, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_realtime_events_task
                    ON realtime_inspection_events(task_id, detected_at DESC);
                """
            )
            missing_jobs = connection.execute(
                """
                SELECT task.* FROM monitoring_tasks AS task
                LEFT JOIN monitoring_jobs AS job ON job.task_id = task.id
                WHERE job.task_id IS NULL
                """
            ).fetchall()
            for task in missing_jobs:
                config = _json_load(task["config_json"])
                connection.execute(
                    """
                    INSERT INTO monitoring_jobs(
                        task_id, source_id, status, started_at, ends_at,
                        segment_seconds, last_processed_at, last_error,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task["id"],
                        task["source_id"],
                        MONITORING_TASK_TO_JOB_STATUS[task["status"]],
                        task["start_time"],
                        task["end_time"],
                        float(config.get("capture_duration_seconds") or 60.0),
                        task["last_run_ended_at"],
                        task["last_error_message"],
                        task["created_at"],
                        task["updated_at"],
                    ),
                )
            self._ensure_column(connection, "detection_runs", "line_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(
                connection, "detection_runs", "source_started_at", "TEXT NOT NULL DEFAULT ''"
            )
            self._ensure_column(
                connection, "detection_runs", "source_ended_at", "TEXT NOT NULL DEFAULT ''"
            )
            self._ensure_column(
                connection,
                "detection_runs",
                "review_status",
                "TEXT NOT NULL DEFAULT 'unreviewed'",
            )
            self._ensure_column(
                connection, "detection_runs", "review_note", "TEXT NOT NULL DEFAULT ''"
            )
            self._ensure_column(
                connection, "detection_runs", "reviewer", "TEXT NOT NULL DEFAULT ''"
            )
            self._ensure_column(
                connection, "detection_runs", "reviewed_at", "TEXT NOT NULL DEFAULT ''"
            )
            for column, definition in (
                ("line_id", "TEXT NOT NULL DEFAULT ''"),
                ("last_seen_at", "TEXT NOT NULL DEFAULT ''"),
                ("event_status", "TEXT NOT NULL DEFAULT 'closed'"),
                ("hit_count", "INTEGER NOT NULL DEFAULT 0"),
                ("class_counts_json", "TEXT NOT NULL DEFAULT '{}'"),
                ("max_confidence", "REAL NOT NULL DEFAULT 0"),
                ("alarm_report", "TEXT NOT NULL DEFAULT ''"),
                ("llm_summary", "TEXT NOT NULL DEFAULT ''"),
                ("updated_at", "TEXT NOT NULL DEFAULT ''"),
            ):
                self._ensure_column(
                    connection, "realtime_inspection_events", column, definition
                )
            connection.execute(
                "UPDATE realtime_inspection_events "
                "SET last_seen_at = COALESCE(NULLIF(last_seen_at, ''), ended_at), "
                "max_confidence = CASE WHEN max_confidence > 0 THEN max_confidence ELSE confidence END, "
                "hit_count = CASE WHEN hit_count > 0 THEN hit_count ELSE 1 END, "
                "updated_at = COALESCE(NULLIF(updated_at, ''), created_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_detection_line_created "
                "ON detection_runs(line_id, created_at DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_detection_source_time "
                "ON detection_runs(source_started_at, source_ended_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_detection_review_status "
                "ON detection_runs(review_status, created_at DESC)"
            )

    @staticmethod
    def _monitoring_task_from_row(row: sqlite3.Row) -> MonitoringTaskRecord:
        return MonitoringTaskRecord(
            id=row["id"],
            session_id=row["session_id"],
            source_id=row["source_id"],
            line_id=row["line_id"],
            zone_id=row["zone_id"],
            status=row["status"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            config=_json_load(row["config_json"]),
            runs_completed=int(row["runs_completed"]),
            runs_succeeded=int(row["runs_succeeded"]),
            runs_failed=int(row["runs_failed"]),
            consecutive_failures=int(row["consecutive_failures"]),
            last_run_started_at=row["last_run_started_at"],
            last_run_ended_at=row["last_run_ended_at"],
            last_detection_id=row["last_detection_id"],
            last_alarm_id=row["last_alarm_id"],
            last_risk_level=row["last_risk_level"],
            last_error_code=row["last_error_code"],
            last_error_message=row["last_error_message"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _monitoring_run_from_row(row: sqlite3.Row) -> MonitoringRunRecord:
        return MonitoringRunRecord(
            id=int(row["id"]),
            task_id=row["task_id"],
            run_index=int(row["run_index"]),
            status=row["status"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            detection_id=row["detection_id"],
            alarm_id=row["alarm_id"],
            risk_level=row["risk_level"],
            error_code=row["error_code"],
            error_message=row["error_message"],
            result=_json_load(row["result_json"]),
        )

    @staticmethod
    def _monitoring_job_from_row(row: sqlite3.Row) -> MonitoringJobRecord:
        return MonitoringJobRecord(
            task_id=row["task_id"],
            source_id=row["source_id"],
            status=row["status"],
            started_at=row["started_at"],
            ends_at=row["ends_at"],
            segment_seconds=float(row["segment_seconds"]),
            last_processed_at=row["last_processed_at"],
            last_error=row["last_error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _stream_segment_from_row(row: sqlite3.Row) -> StreamSegmentRecord:
        return StreamSegmentRecord(
            segment_id=row["segment_id"],
            task_id=row["task_id"],
            source_id=row["source_id"],
            video_path=row["video_path"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            status=row["status"],
            detection_id=row["detection_id"],
            retry_count=int(row["retry_count"]),
            created_at=row["created_at"],
        )

    @staticmethod
    def _realtime_task_from_row(row: sqlite3.Row) -> RealtimeInspectionTaskRecord:
        return RealtimeInspectionTaskRecord(
            id=row["id"], session_id=row["session_id"], source_id=row["source_id"],
            line_id=row["line_id"], zone_id=row["zone_id"], start_time=row["start_time"],
            end_time=row["end_time"], status=row["status"], sample_fps=float(row["sample_fps"]),
            created_at=row["created_at"], updated_at=row["updated_at"],
            started_at=row["started_at"], stopped_at=row["stopped_at"],
            frames_read=int(row["frames_read"]), frames_inferred=int(row["frames_inferred"]),
            frames_dropped=int(row["frames_dropped"]), inference_failures=int(row["inference_failures"]),
            reconnect_count=int(row["reconnect_count"]), events_detected=int(row["events_detected"]),
            alarms_created=int(row["alarms_created"]), highest_risk_level=row["highest_risk_level"],
            last_frame_at=row["last_frame_at"], last_inference_at=row["last_inference_at"],
            latest_detection_id=row["latest_detection_id"], latest_alarm_id=row["latest_alarm_id"],
            latest_event_frame=row["latest_event_frame"], last_error_code=row["last_error_code"],
            last_error_message=row["last_error_message"], config=_json_load(row["config_json"]),
        )

    @staticmethod
    def _realtime_event_from_row(row: sqlite3.Row) -> RealtimeInspectionEventRecord:
        return RealtimeInspectionEventRecord(
            event_id=row["event_id"], task_id=row["task_id"], source_id=row["source_id"],
            detected_at=row["detected_at"], ended_at=row["ended_at"],
            class_name=row["class_name"], confidence=float(row["confidence"]),
            bbox=list(_json_load(row["bbox_json"])), risk_level=row["risk_level"],
            detection_id=row["detection_id"], alarm_id=row["alarm_id"], image_path=row["image_path"],
            metadata=_json_load(row["metadata_json"]), created_at=row["created_at"],
            line_id=row["line_id"], last_seen_at=row["last_seen_at"],
            event_status=row["event_status"], hit_count=int(row["hit_count"]),
            class_counts=dict(_json_load(row["class_counts_json"])),
            max_confidence=float(row["max_confidence"]),
            alarm_report=row["alarm_report"], llm_summary=row["llm_summary"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        columns = {
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def ensure_session(self, session_id: str) -> None:
        session_id = session_id.strip()
        if not session_id:
            raise ValueError("session_id 不能为空")
        timestamp = self._timestamp()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions(id, created_at, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (session_id, timestamp, timestamp),
            )

    def record_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        intent: str = "",
        tool_name: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"不支持的消息角色：{role}")
        self.ensure_session(session_id)
        timestamp = self._timestamp()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO messages(
                    session_id, role, content, intent, tool_name,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    content,
                    intent,
                    tool_name,
                    _json_dump(metadata or {}),
                    timestamp,
                ),
            )
            return int(cursor.lastrowid)

    def list_messages(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        if limit < 1:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content, intent, tool_name, metadata_json, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [
            {
                "role": row["role"],
                "content": row["content"],
                "intent": row["intent"],
                "tool_name": row["tool_name"],
                "metadata": _json_load(row["metadata_json"]),
                "created_at": row["created_at"],
            }
            for row in reversed(rows)
        ]

    def record_detection(
        self,
        session_id: str,
        *,
        source_type: str,
        source_path: str,
        detection: Dict[str, Any],
        alarm_document: Dict[str, Any],
        alarm_report: str,
        line_id: str = "",
        source_started_at: str = "",
        source_ended_at: str = "",
    ) -> tuple[DetectionRecord, AlarmRecord]:
        self.ensure_session(session_id)
        timestamp = self._timestamp()
        detection_id = f"det-{uuid.uuid4().hex[:12]}"
        alarm_id = str(alarm_document.get("report_id") or f"alarm-{uuid.uuid4().hex[:12]}")
        overall_risk = alarm_document.get("overall_risk") or {}
        risk_level = str(overall_risk.get("level") or "none").lower()
        status = "pending" if risk_level != "none" else "inactive"
        requires_stop = bool(overall_risk.get("requires_stop"))

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO detection_runs(
                    id, session_id, source_type, source_path, status,
                    risk_level, summary_json, alarm_report, line_id,
                    source_started_at, source_ended_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    detection_id,
                    session_id,
                    source_type,
                    source_path,
                    str(detection.get("status") or "completed"),
                    risk_level,
                    _json_dump(detection),
                    alarm_report,
                    str(line_id or "").strip(),
                    str(source_started_at or "").strip(),
                    str(source_ended_at or "").strip(),
                    timestamp,
                ),
            )
            connection.execute(
                """
                INSERT INTO alarms(
                    id, detection_id, session_id, risk_level, status,
                    requires_stop, report_json, report_text, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alarm_id,
                    detection_id,
                    session_id,
                    risk_level,
                    status,
                    int(requires_stop),
                    _json_dump(alarm_document),
                    alarm_report,
                    timestamp,
                    timestamp,
                ),
            )

        detection_record = DetectionRecord(
            id=detection_id,
            session_id=session_id,
            source_type=source_type,
            source_path=source_path,
            status=str(detection.get("status") or "completed"),
            risk_level=risk_level,
            summary=detection,
            alarm_report=alarm_report,
            line_id=str(line_id or "").strip(),
            source_started_at=str(source_started_at or "").strip(),
            source_ended_at=str(source_ended_at or "").strip(),
            review_status="unreviewed",
            review_note="",
            reviewer="",
            reviewed_at="",
            created_at=timestamp,
        )
        alarm_record = AlarmRecord(
            id=alarm_id,
            detection_id=detection_id,
            session_id=session_id,
            risk_level=risk_level,
            status=status,
            requires_stop=requires_stop,
            report=alarm_document,
            report_text=alarm_report,
            created_at=timestamp,
            updated_at=timestamp,
        )
        return detection_record, alarm_record

    @staticmethod
    def _detection_from_row(row: sqlite3.Row) -> DetectionRecord:
        return DetectionRecord(
            id=row["id"],
            session_id=row["session_id"],
            source_type=row["source_type"],
            source_path=row["source_path"],
            status=row["status"],
            risk_level=row["risk_level"],
            summary=_json_load(row["summary_json"]),
            alarm_report=row["alarm_report"],
            line_id=row["line_id"],
            source_started_at=row["source_started_at"],
            source_ended_at=row["source_ended_at"],
            review_status=row["review_status"],
            review_note=row["review_note"],
            reviewer=row["reviewer"],
            reviewed_at=row["reviewed_at"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _alarm_from_row(row: sqlite3.Row) -> AlarmRecord:
        return AlarmRecord(
            id=row["id"],
            detection_id=row["detection_id"],
            session_id=row["session_id"],
            risk_level=row["risk_level"],
            status=row["status"],
            requires_stop=bool(row["requires_stop"]),
            report=_json_load(row["report_json"]),
            report_text=row["report_text"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def latest_detection(self, session_id: Optional[str] = None) -> Optional[DetectionRecord]:
        query = "SELECT * FROM detection_runs"
        parameters: tuple[Any, ...] = ()
        if session_id:
            query += " WHERE session_id = ?"
            parameters = (session_id,)
        query += " ORDER BY created_at DESC, rowid DESC LIMIT 1"
        with self._connect() as connection:
            row = connection.execute(query, parameters).fetchone()
        return self._detection_from_row(row) if row else None

    def get_detection(self, detection_id: str) -> Optional[DetectionRecord]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM detection_runs WHERE id = ?", (detection_id,)
            ).fetchone()
        return self._detection_from_row(row) if row else None

    def query_detections(
        self,
        *,
        start_time: str = "",
        end_time: str = "",
        risk_level: str = "",
        line_id: str = "",
        source_type: str = "",
        review_status: str = "",
        limit: int = 100,
    ) -> List[DetectionRecord]:
        if limit < 1:
            return []
        query = "SELECT * FROM detection_runs WHERE 1 = 1"
        parameters: List[Any] = []
        if start_time:
            query += (
                " AND julianday(COALESCE(NULLIF(source_ended_at, ''), "
                "NULLIF(source_started_at, ''), created_at)) >= julianday(?)"
            )
            parameters.append(start_time)
        if end_time:
            query += (
                " AND julianday(COALESCE(NULLIF(source_started_at, ''), created_at)) "
                "<= julianday(?)"
            )
            parameters.append(end_time)
        for column, value in (
            ("risk_level", risk_level.lower()),
            ("line_id", line_id),
            ("source_type", source_type.lower()),
            ("review_status", review_status.lower()),
        ):
            if value:
                query += f" AND {column} = ?"
                parameters.append(value)
        query += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return [self._detection_from_row(row) for row in rows]

    def set_detection_review(
        self,
        detection_id: str,
        session_id: str,
        action: str,
        *,
        reviewer: str = "",
        note: str = "",
    ) -> DetectionRecord:
        action = action.lower().strip()
        if action not in VALID_REVIEW_ACTIONS:
            raise ValueError(f"不支持的复核动作：{action}")
        self.ensure_session(session_id)
        timestamp = self._timestamp()
        status_by_action = {
            "confirm": "confirmed",
            "reject": "rejected",
            "close": "closed",
            "reopen": "unreviewed",
        }
        new_status = status_by_action[action]
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id FROM detection_runs WHERE id = ?", (detection_id,)
            ).fetchone()
            if row is None:
                raise LookupError(f"找不到检测记录：{detection_id}")
            connection.execute(
                """
                UPDATE detection_runs
                SET review_status = ?, review_note = ?, reviewer = ?, reviewed_at = ?
                WHERE id = ?
                """,
                (new_status, note.strip(), reviewer.strip(), timestamp, detection_id),
            )
            connection.execute(
                """
                INSERT INTO detection_review_actions(
                    detection_id, session_id, action, reviewer, note, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    detection_id,
                    session_id,
                    action,
                    reviewer.strip(),
                    note.strip(),
                    timestamp,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM detection_runs WHERE id = ?", (detection_id,)
            ).fetchone()
        return self._detection_from_row(updated)

    def list_detection_review_actions(self, detection_id: str) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT session_id, action, reviewer, note, created_at
                FROM detection_review_actions
                WHERE detection_id = ?
                ORDER BY id ASC
                """,
                (detection_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_alarm(self, alarm_id: str) -> Optional[AlarmRecord]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM alarms WHERE id = ?", (alarm_id,)
            ).fetchone()
        return self._alarm_from_row(row) if row else None

    def get_alarm_for_detection(self, detection_id: str) -> Optional[AlarmRecord]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM alarms WHERE detection_id = ?", (detection_id,)
            ).fetchone()
        return self._alarm_from_row(row) if row else None

    def get_alarms_for_detections(
        self, detection_ids: List[str]
    ) -> Dict[str, AlarmRecord]:
        normalized = list(dict.fromkeys(
            str(detection_id).strip()
            for detection_id in detection_ids
            if str(detection_id).strip()
        ))
        result: Dict[str, AlarmRecord] = {}
        with self._connect() as connection:
            for offset in range(0, len(normalized), 900):
                chunk = normalized[offset : offset + 900]
                if not chunk:
                    continue
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    f"SELECT * FROM alarms WHERE detection_id IN ({placeholders})",
                    tuple(chunk),
                ).fetchall()
                for row in rows:
                    alarm = self._alarm_from_row(row)
                    result[alarm.detection_id] = alarm
        return result

    def list_alarms(
        self, *, status: str = "", limit: int = 500
    ) -> List[AlarmRecord]:
        if limit < 1:
            return []
        normalized_status = str(status or "").strip().lower()
        if normalized_status and normalized_status not in VALID_ALARM_STATUSES:
            raise ValueError(f"不支持的报警状态：{status}")
        query = "SELECT * FROM alarms"
        parameters: List[Any] = []
        if normalized_status:
            query += " WHERE status = ?"
            parameters.append(normalized_status)
        query += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        parameters.append(min(int(limit), 500))
        with self._connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return [self._alarm_from_row(row) for row in rows]

    def current_alarm(
        self,
        session_id: Optional[str] = None,
        *,
        line_id: str = "",
    ) -> Optional[AlarmRecord]:
        query = (
            "SELECT a.* FROM alarms a "
            "JOIN detection_runs d ON d.id = a.detection_id "
            "WHERE a.status != 'inactive'"
        )
        parameters: List[Any] = []
        if session_id:
            query += " AND a.session_id = ?"
            parameters.append(session_id)
        if line_id:
            query += " AND d.line_id = ?"
            parameters.append(line_id)
        query += (
            " ORDER BY CASE WHEN a.status = 'pending' THEN 0 ELSE 1 END, "
            "a.created_at DESC, a.rowid DESC LIMIT 1"
        )
        with self._connect() as connection:
            row = connection.execute(query, tuple(parameters)).fetchone()
        return self._alarm_from_row(row) if row else None

    def latest_actionable_alarm(self, session_id: Optional[str] = None) -> Optional[AlarmRecord]:
        query = "SELECT * FROM alarms WHERE status != 'inactive'"
        parameters: tuple[Any, ...] = ()
        if session_id:
            query += " AND session_id = ?"
            parameters = (session_id,)
        query += " ORDER BY created_at DESC, rowid DESC LIMIT 1"
        with self._connect() as connection:
            row = connection.execute(query, parameters).fetchone()
        return self._alarm_from_row(row) if row else None

    def set_alarm_action(
        self,
        alarm_id: str,
        session_id: str,
        action: str,
        note: str = "",
    ) -> AlarmRecord:
        if action not in VALID_ALARM_ACTIONS:
            raise ValueError(f"不支持的报警动作：{action}")
        self.ensure_session(session_id)
        timestamp = self._timestamp()
        new_status = "confirmed" if action == "confirm" else "cancelled"
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM alarms WHERE id = ?", (alarm_id,)
            ).fetchone()
            if row is None:
                raise LookupError(f"找不到报警：{alarm_id}")
            if row["status"] == "inactive":
                raise ValueError("无风险记录不需要确认或取消")
            connection.execute(
                "UPDATE alarms SET status = ?, updated_at = ? WHERE id = ?",
                (new_status, timestamp, alarm_id),
            )
            connection.execute(
                """
                INSERT INTO alarm_actions(alarm_id, session_id, action, note, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (alarm_id, session_id, action, note, timestamp),
            )
            updated = connection.execute(
                "SELECT * FROM alarms WHERE id = ?", (alarm_id,)
            ).fetchone()
        return self._alarm_from_row(updated)

    def set_pending_alarm_actions(
        self,
        alarm_ids: List[str],
        session_id: str,
        action: str,
        note: str = "",
    ) -> Dict[str, List[str]]:
        """Atomically confirm or cancel multiple pending alarms.

        Existing terminal alarm states are intentionally left unchanged so a bulk
        operation cannot silently reverse a previous human decision.
        """
        action = action.lower().strip()
        if action not in VALID_ALARM_ACTIONS:
            raise ValueError(f"不支持的报警动作：{action}")
        normalized_ids = list(dict.fromkeys(
            str(alarm_id).strip() for alarm_id in alarm_ids if str(alarm_id).strip()
        ))
        if not normalized_ids:
            return {"updated": [], "unchanged": [], "missing": []}
        if len(normalized_ids) > 500:
            raise ValueError("单次最多处理 500 条报警")

        self.ensure_session(session_id)
        timestamp = self._timestamp()
        new_status = "confirmed" if action == "confirm" else "cancelled"
        updated_ids: List[str] = []
        unchanged_ids: List[str] = []
        missing_ids: List[str] = []
        with self._connect() as connection:
            for alarm_id in normalized_ids:
                row = connection.execute(
                    "SELECT id, status FROM alarms WHERE id = ?", (alarm_id,)
                ).fetchone()
                if row is None:
                    missing_ids.append(alarm_id)
                    continue
                if str(row["status"]) != "pending":
                    unchanged_ids.append(alarm_id)
                    continue
                connection.execute(
                    "UPDATE alarms SET status = ?, updated_at = ? WHERE id = ?",
                    (new_status, timestamp, alarm_id),
                )
                connection.execute(
                    """
                    INSERT INTO alarm_actions(alarm_id, session_id, action, note, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (alarm_id, session_id, action, note.strip(), timestamp),
                )
                updated_ids.append(alarm_id)
        return {
            "updated": updated_ids,
            "unchanged": unchanged_ids,
            "missing": missing_ids,
        }

    def count_risk_level(self, risk_level: str, target_date: date) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM alarms
                WHERE risk_level = ? AND substr(created_at, 1, 10) = ?
                """,
                (risk_level.lower(), target_date.isoformat()),
            ).fetchone()
        return int(row["count"])

    def daily_summary(self, target_date: date) -> Dict[str, Any]:
        day_text = target_date.isoformat()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT risk_level, status, COUNT(*) AS count
                FROM alarms
                WHERE substr(created_at, 1, 10) = ?
                GROUP BY risk_level, status
                """,
                (day_text,),
            ).fetchall()
            detections = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM detection_runs
                WHERE substr(created_at, 1, 10) = ?
                """,
                (day_text,),
            ).fetchone()

        risk_counts = {level: 0 for level in ("none", "low", "medium", "high")}
        status_counts = {status: 0 for status in VALID_ALARM_STATUSES}
        for row in rows:
            risk_counts[row["risk_level"]] = risk_counts.get(row["risk_level"], 0) + int(
                row["count"]
            )
            status_counts[row["status"]] = status_counts.get(row["status"], 0) + int(
                row["count"]
            )
        return {
            "date": day_text,
            "detection_count": int(detections["count"]),
            "alarm_count": sum(count for level, count in risk_counts.items() if level != "none"),
            "risk_counts": risk_counts,
            "status_counts": status_counts,
        }

    def filtered_summary(
        self,
        *,
        start_time: str = "",
        end_time: str = "",
        risk_level: str = "",
        line_id: str = "",
        source_type: str = "",
        review_status: str = "",
        limit: int = 10000,
    ) -> Dict[str, Any]:
        records = self.query_detections(
            start_time=start_time,
            end_time=end_time,
            risk_level=risk_level,
            line_id=line_id,
            source_type=source_type,
            review_status=review_status,
            limit=limit,
        )
        risk_counts = {level: 0 for level in ("none", "low", "medium", "high")}
        source_counts: Dict[str, int] = {}
        review_counts = {status: 0 for status in VALID_REVIEW_STATUSES}
        class_counts: Dict[str, int] = {}
        alarm_status_counts = {status: 0 for status in VALID_ALARM_STATUSES}
        alarm_status_by_detection: Dict[str, str] = {}
        detection_ids = [record.id for record in records]
        with self._connect() as connection:
            for offset in range(0, len(detection_ids), 900):
                chunk = detection_ids[offset : offset + 900]
                if not chunk:
                    continue
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    f"SELECT detection_id, status FROM alarms "
                    f"WHERE detection_id IN ({placeholders})",
                    tuple(chunk),
                ).fetchall()
                alarm_status_by_detection.update(
                    {str(row["detection_id"]): str(row["status"]) for row in rows}
                )
        for record in records:
            risk_counts[record.risk_level] = risk_counts.get(record.risk_level, 0) + 1
            source_counts[record.source_type] = source_counts.get(record.source_type, 0) + 1
            review_counts[record.review_status] = review_counts.get(record.review_status, 0) + 1
            for class_name, count in (record.summary.get("class_counts") or {}).items():
                class_counts[str(class_name)] = class_counts.get(str(class_name), 0) + int(count)
            alarm_status = alarm_status_by_detection.get(record.id)
            if alarm_status:
                alarm_status_counts[alarm_status] = (
                    alarm_status_counts.get(alarm_status, 0) + 1
                )
        return {
            "start_time": start_time,
            "end_time": end_time,
            "line_id": line_id,
            "risk_level": risk_level,
            "source_type": source_type,
            "review_status": review_status,
            "detection_count": len(records),
            "alarm_count": sum(
                count for level, count in risk_counts.items() if level != "none"
            ),
            "risk_counts": risk_counts,
            "source_counts": source_counts,
            "review_counts": review_counts,
            "alarm_status_counts": alarm_status_counts,
            "class_counts": class_counts,
        }

    def create_monitoring_task(
        self,
        session_id: str,
        *,
        source_id: str,
        line_id: str,
        zone_id: str,
        start_time: str,
        end_time: str,
        config: Dict[str, Any],
    ) -> MonitoringTaskRecord:
        self.ensure_session(session_id)
        timestamp = self._timestamp()
        task_id = f"monitor-{uuid.uuid4().hex[:12]}"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO monitoring_tasks(
                    id, session_id, source_id, line_id, zone_id, status,
                    start_time, end_time, config_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'scheduled', ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    session_id,
                    source_id,
                    line_id,
                    zone_id,
                    start_time,
                    end_time,
                    _json_dump(config),
                    timestamp,
                    timestamp,
                ),
            )
            connection.execute(
                """
                INSERT INTO monitoring_jobs(
                    task_id, source_id, status, started_at, ends_at,
                    segment_seconds, created_at, updated_at
                ) VALUES (?, ?, 'pending', ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    source_id,
                    start_time,
                    end_time,
                    float(config.get("capture_duration_seconds") or 60.0),
                    timestamp,
                    timestamp,
                ),
            )
            row = connection.execute(
                "SELECT * FROM monitoring_tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return self._monitoring_task_from_row(row)

    def get_monitoring_task(self, task_id: str) -> Optional[MonitoringTaskRecord]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM monitoring_tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return self._monitoring_task_from_row(row) if row else None

    def list_monitoring_tasks(
        self,
        *,
        session_id: str = "",
        source_id: str = "",
        statuses: tuple[str, ...] = (),
        limit: int = 20,
    ) -> List[MonitoringTaskRecord]:
        if limit < 1:
            return []
        query = "SELECT * FROM monitoring_tasks WHERE 1 = 1"
        parameters: List[Any] = []
        if session_id:
            query += " AND session_id = ?"
            parameters.append(session_id)
        if source_id:
            query += " AND source_id = ?"
            parameters.append(source_id)
        if statuses:
            unknown = set(statuses) - VALID_MONITORING_STATUSES
            if unknown:
                raise ValueError(f"无效监控任务状态：{', '.join(sorted(unknown))}")
            placeholders = ",".join("?" for _ in statuses)
            query += f" AND status IN ({placeholders})"
            parameters.extend(statuses)
        query += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return [self._monitoring_task_from_row(row) for row in rows]

    def update_monitoring_task(
        self,
        task_id: str,
        **changes: Any,
    ) -> MonitoringTaskRecord:
        allowed = {
            "status",
            "last_run_started_at",
            "last_run_ended_at",
            "last_detection_id",
            "last_alarm_id",
            "last_risk_level",
            "last_error_code",
            "last_error_message",
        }
        unknown = sorted(set(changes) - allowed)
        if unknown:
            raise ValueError(f"不支持的监控任务更新字段：{', '.join(unknown)}")
        if changes.get("status") and changes["status"] not in VALID_MONITORING_STATUSES:
            raise ValueError(f"无效监控任务状态：{changes['status']}")
        if not changes:
            task = self.get_monitoring_task(task_id)
            if task is None:
                raise LookupError(f"找不到监控任务：{task_id}")
            return task
        changes["updated_at"] = self._timestamp()
        assignments = ", ".join(f"{name} = ?" for name in changes)
        parameters = [changes[name] for name in changes]
        parameters.append(task_id)
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE monitoring_tasks SET {assignments} WHERE id = ?",
                tuple(parameters),
            )
            if cursor.rowcount == 0:
                raise LookupError(f"找不到监控任务：{task_id}")
            row = connection.execute(
                "SELECT * FROM monitoring_tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if "status" in changes:
                job_status = MONITORING_TASK_TO_JOB_STATUS[str(row["status"])]
                job_error = (
                    str(changes.get("last_error_message") or "")
                    if job_status == "failed"
                    else None
                )
                if job_error is None:
                    connection.execute(
                        "UPDATE monitoring_jobs SET status = ?, updated_at = ? "
                        "WHERE task_id = ?",
                        (job_status, changes["updated_at"], task_id),
                    )
                else:
                    connection.execute(
                        "UPDATE monitoring_jobs SET status = ?, last_error = ?, "
                        "updated_at = ? WHERE task_id = ?",
                        (job_status, job_error, changes["updated_at"], task_id),
                    )
        return self._monitoring_task_from_row(row)

    def request_monitoring_task_stop(self, task_id: str) -> MonitoringTaskRecord:
        """Atomically request stop without moving a terminal task backwards."""
        timestamp = self._timestamp()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE monitoring_tasks
                SET status = 'stop_requested', updated_at = ?
                WHERE id = ? AND status IN ('scheduled', 'running')
                """,
                (timestamp, task_id),
            )
            row = connection.execute(
                "SELECT * FROM monitoring_tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is not None:
                connection.execute(
                    "UPDATE monitoring_jobs SET status = ?, updated_at = ? "
                    "WHERE task_id = ?",
                    (
                        MONITORING_TASK_TO_JOB_STATUS[str(row["status"])],
                        timestamp,
                        task_id,
                    ),
                )
        if row is None:
            raise LookupError(f"找不到监控任务：{task_id}")
        return self._monitoring_task_from_row(row)

    def interrupt_monitoring_task_if_active(
        self,
        task_id: str,
        *,
        error_code: str,
        error_message: str,
    ) -> MonitoringTaskRecord:
        """Atomically interrupt only a task that has not already terminated."""
        timestamp = self._timestamp()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE monitoring_tasks
                SET status = 'interrupted', last_error_code = ?,
                    last_error_message = ?, updated_at = ?
                WHERE id = ? AND status IN ('scheduled', 'running', 'stop_requested')
                """,
                (error_code, error_message, timestamp, task_id),
            )
            row = connection.execute(
                "SELECT * FROM monitoring_tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is not None and str(row["status"]) == "interrupted":
                connection.execute(
                    """
                    UPDATE monitoring_jobs
                    SET status = 'failed', last_error = ?, updated_at = ?
                    WHERE task_id = ?
                    """,
                    (error_message, timestamp, task_id),
                )
        if row is None:
            raise LookupError(f"找不到监控任务：{task_id}")
        return self._monitoring_task_from_row(row)

    def record_monitoring_run(
        self,
        task_id: str,
        *,
        succeeded: bool,
        started_at: str,
        ended_at: str,
        detection_id: str = "",
        alarm_id: str = "",
        risk_level: str = "",
        error_code: str = "",
        error_message: str = "",
        result: Optional[Dict[str, Any]] = None,
    ) -> tuple[MonitoringTaskRecord, MonitoringRunRecord]:
        timestamp = self._timestamp()
        with self._connect() as connection:
            task_row = connection.execute(
                "SELECT * FROM monitoring_tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if task_row is None:
                raise LookupError(f"找不到监控任务：{task_id}")
            run_index = int(task_row["runs_completed"]) + 1
            runs_succeeded = int(task_row["runs_succeeded"]) + int(succeeded)
            runs_failed = int(task_row["runs_failed"]) + int(not succeeded)
            consecutive_failures = (
                0 if succeeded else int(task_row["consecutive_failures"]) + 1
            )
            cursor = connection.execute(
                """
                INSERT INTO monitoring_task_runs(
                    task_id, run_index, status, started_at, ended_at,
                    detection_id, alarm_id, risk_level, error_code,
                    error_message, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    run_index,
                    "succeeded" if succeeded else "failed",
                    started_at,
                    ended_at,
                    detection_id,
                    alarm_id,
                    risk_level,
                    error_code,
                    error_message,
                    _json_dump(result or {}),
                ),
            )
            connection.execute(
                """
                UPDATE monitoring_tasks SET
                    runs_completed = ?, runs_succeeded = ?, runs_failed = ?,
                    consecutive_failures = ?, last_run_started_at = ?,
                    last_run_ended_at = ?, last_detection_id = ?,
                    last_alarm_id = ?, last_risk_level = ?, last_error_code = ?,
                    last_error_message = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    run_index,
                    runs_succeeded,
                    runs_failed,
                    consecutive_failures,
                    started_at,
                    ended_at,
                    detection_id,
                    alarm_id,
                    risk_level,
                    error_code,
                    error_message,
                    timestamp,
                    task_id,
                ),
            )
            updated_task = connection.execute(
                "SELECT * FROM monitoring_tasks WHERE id = ?", (task_id,)
            ).fetchone()
            run_row = connection.execute(
                "SELECT * FROM monitoring_task_runs WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
            connection.execute(
                """
                UPDATE monitoring_jobs
                SET last_processed_at = ?, last_error = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (ended_at, error_message, timestamp, task_id),
            )
        return (
            self._monitoring_task_from_row(updated_task),
            self._monitoring_run_from_row(run_row),
        )

    def list_monitoring_runs(
        self,
        task_id: str,
        *,
        limit: int = 20,
    ) -> List[MonitoringRunRecord]:
        if limit < 1:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM monitoring_task_runs
                WHERE task_id = ?
                ORDER BY run_index DESC
                LIMIT ?
                """,
                (task_id, limit),
            ).fetchall()
        return [self._monitoring_run_from_row(row) for row in rows]

    def get_monitoring_job(self, task_id: str) -> Optional[MonitoringJobRecord]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM monitoring_jobs WHERE task_id = ?", (task_id,)
            ).fetchone()
        return self._monitoring_job_from_row(row) if row else None

    def list_monitoring_jobs(
        self,
        *,
        source_id: str = "",
        statuses: tuple[str, ...] = (),
        limit: int = 20,
    ) -> List[MonitoringJobRecord]:
        if limit < 1:
            return []
        unknown = set(statuses) - VALID_MONITORING_JOB_STATUSES
        if unknown:
            raise ValueError(f"无效监控运行状态：{', '.join(sorted(unknown))}")
        query = "SELECT * FROM monitoring_jobs WHERE 1 = 1"
        parameters: List[Any] = []
        if source_id:
            query += " AND source_id = ?"
            parameters.append(source_id)
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            query += f" AND status IN ({placeholders})"
            parameters.extend(statuses)
        query += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return [self._monitoring_job_from_row(row) for row in rows]

    def update_monitoring_job(
        self,
        task_id: str,
        **changes: Any,
    ) -> MonitoringJobRecord:
        allowed = {"status", "last_processed_at", "last_error"}
        unknown = sorted(set(changes) - allowed)
        if unknown:
            raise ValueError(f"不支持的监控运行状态字段：{', '.join(unknown)}")
        if changes.get("status") and changes["status"] not in VALID_MONITORING_JOB_STATUSES:
            raise ValueError(f"无效监控运行状态：{changes['status']}")
        if not changes:
            job = self.get_monitoring_job(task_id)
            if job is None:
                raise LookupError(f"找不到监控运行状态：{task_id}")
            return job
        changes["updated_at"] = self._timestamp()
        assignments = ", ".join(f"{name} = ?" for name in changes)
        parameters = [changes[name] for name in changes]
        parameters.append(task_id)
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE monitoring_jobs SET {assignments} WHERE task_id = ?",
                tuple(parameters),
            )
            if cursor.rowcount == 0:
                raise LookupError(f"找不到监控运行状态：{task_id}")
            row = connection.execute(
                "SELECT * FROM monitoring_jobs WHERE task_id = ?", (task_id,)
            ).fetchone()
        return self._monitoring_job_from_row(row)

    def claim_stream_segment(
        self,
        task_id: str,
        *,
        source_id: str,
        started_at: str | datetime,
        ended_at: str | datetime,
    ) -> tuple[StreamSegmentRecord, bool]:
        """Create or atomically claim one logical stream window.

        The unique logical window prevents two workers, retries, or restarts from
        detecting the same segment concurrently or after it has completed.
        """
        normalized_start = _normalized_utc_time(started_at, "started_at")
        normalized_end = _normalized_utc_time(ended_at, "ended_at")
        if normalized_end <= normalized_start:
            raise ValueError("ended_at 必须晚于 started_at")
        identity = f"{task_id}|{source_id}|{normalized_start}|{normalized_end}"
        segment_id = f"segment-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]}"
        timestamp = self._timestamp()
        with self._connect() as connection:
            job = connection.execute(
                "SELECT source_id FROM monitoring_jobs WHERE task_id = ?", (task_id,)
            ).fetchone()
            if job is None:
                raise LookupError(f"找不到监控运行状态：{task_id}")
            if str(job["source_id"]) != source_id:
                raise ValueError("stream segment 的 source_id 与监控任务不一致")
            connection.execute(
                """
                INSERT INTO stream_segments(
                    segment_id, task_id, source_id, started_at, ended_at,
                    status, created_at
                ) VALUES (?, ?, ?, ?, ?, 'pending', ?)
                ON CONFLICT(task_id, source_id, started_at, ended_at) DO NOTHING
                """,
                (
                    segment_id,
                    task_id,
                    source_id,
                    normalized_start,
                    normalized_end,
                    timestamp,
                ),
            )
            row = connection.execute(
                """
                SELECT * FROM stream_segments
                WHERE task_id = ? AND source_id = ?
                  AND started_at = ? AND ended_at = ?
                """,
                (task_id, source_id, normalized_start, normalized_end),
            ).fetchone()
            should_process = str(row["status"]) in {"pending", "failed"}
            if should_process:
                retry_count = int(row["retry_count"]) + int(row["status"] == "failed")
                connection.execute(
                    """
                    UPDATE stream_segments
                    SET status = 'processing', retry_count = ?
                    WHERE segment_id = ? AND status IN ('pending', 'failed')
                    """,
                    (retry_count, row["segment_id"]),
                )
                row = connection.execute(
                    "SELECT * FROM stream_segments WHERE segment_id = ?",
                    (row["segment_id"],),
                ).fetchone()
        return self._stream_segment_from_row(row), should_process

    def finish_stream_segment(
        self,
        segment_id: str,
        *,
        succeeded: bool,
        video_path: str = "",
        detection_id: str = "",
    ) -> StreamSegmentRecord:
        with self._connect() as connection:
            current = connection.execute(
                "SELECT * FROM stream_segments WHERE segment_id = ?", (segment_id,)
            ).fetchone()
            if current is None:
                raise LookupError(f"找不到视频片段：{segment_id}")
            if str(current["status"]) != "completed":
                connection.execute(
                    """
                    UPDATE stream_segments
                    SET status = ?, video_path = ?, detection_id = ?
                    WHERE segment_id = ?
                    """,
                    (
                        "completed" if succeeded else "failed",
                        str(video_path or current["video_path"]),
                        str(detection_id or current["detection_id"]),
                        segment_id,
                    ),
                )
            row = connection.execute(
                "SELECT * FROM stream_segments WHERE segment_id = ?", (segment_id,)
            ).fetchone()
        return self._stream_segment_from_row(row)

    def list_stream_segments(
        self,
        task_id: str,
        *,
        limit: int = 100,
    ) -> List[StreamSegmentRecord]:
        if limit < 1:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM stream_segments
                WHERE task_id = ?
                ORDER BY started_at DESC, segment_id DESC
                LIMIT ?
                """,
                (task_id, limit),
            ).fetchall()
        return [self._stream_segment_from_row(row) for row in rows]

    def interrupt_active_monitoring_tasks(self) -> int:
        timestamp = self._timestamp()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE monitoring_tasks
                SET status = 'interrupted',
                    last_error_code = 'process_restarted',
                    last_error_message = '应用进程已重启，任务未自动恢复。',
                    updated_at = ?
                WHERE status IN ('scheduled', 'running', 'stop_requested')
                """,
                (timestamp,),
            )
            connection.execute(
                """
                UPDATE monitoring_jobs
                SET status = 'failed',
                    last_error = '应用进程已重启，任务未自动恢复。',
                    updated_at = ?
                WHERE status IN ('pending', 'connecting', 'running', 'stopping')
                """,
                (timestamp,),
            )
            connection.execute(
                """
                UPDATE stream_segments
                SET status = 'failed'
                WHERE status = 'processing'
                """
            )
            return int(cursor.rowcount)

    @staticmethod
    def _stream_archive_state_from_row(row: sqlite3.Row) -> StreamArchiveStateRecord:
        return StreamArchiveStateRecord(
            source_id=str(row["source_id"]),
            status=str(row["status"]),
            segment_seconds=float(row["segment_seconds"]),
            retention_seconds=float(row["retention_seconds"]),
            last_segment_at=str(row["last_segment_at"]),
            last_error=str(row["last_error"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _stream_archive_segment_from_row(
        row: sqlite3.Row,
    ) -> StreamArchiveSegmentRecord:
        return StreamArchiveSegmentRecord(
            segment_id=str(row["segment_id"]),
            source_id=str(row["source_id"]),
            video_path=str(row["video_path"]),
            started_at=str(row["started_at"]),
            ended_at=str(row["ended_at"]),
            duration_seconds=float(row["duration_seconds"]),
            status=str(row["status"]),
            metadata=dict(_json_load(row["metadata_json"])),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def upsert_stream_archive_state(
        self,
        source_id: str,
        *,
        status: str,
        segment_seconds: float,
        retention_seconds: float,
        last_segment_at: str = "",
        last_error: str = "",
    ) -> StreamArchiveStateRecord:
        if status not in VALID_STREAM_ARCHIVE_STATUSES:
            raise ValueError(f"无效录像归档状态：{status}")
        if float(segment_seconds) < 1:
            raise ValueError("segment_seconds 必须不小于 1")
        if float(retention_seconds) < 3600:
            raise ValueError("retention_seconds 必须不小于 3600")
        timestamp = self._timestamp()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO stream_archive_state(
                    source_id, status, segment_seconds, retention_seconds,
                    last_segment_at, last_error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    status = excluded.status,
                    segment_seconds = excluded.segment_seconds,
                    retention_seconds = excluded.retention_seconds,
                    last_segment_at = CASE
                        WHEN excluded.last_segment_at <> '' THEN excluded.last_segment_at
                        ELSE stream_archive_state.last_segment_at
                    END,
                    last_error = excluded.last_error,
                    updated_at = excluded.updated_at
                """,
                (
                    source_id,
                    status,
                    float(segment_seconds),
                    float(retention_seconds),
                    last_segment_at,
                    last_error,
                    timestamp,
                    timestamp,
                ),
            )
            row = connection.execute(
                "SELECT * FROM stream_archive_state WHERE source_id = ?",
                (source_id,),
            ).fetchone()
        return self._stream_archive_state_from_row(row)

    def update_stream_archive_state(
        self,
        source_id: str,
        **changes: Any,
    ) -> StreamArchiveStateRecord:
        allowed = {"status", "last_segment_at", "last_error"}
        unknown = sorted(set(changes) - allowed)
        if unknown:
            raise ValueError(f"不支持的录像归档状态字段：{', '.join(unknown)}")
        if changes.get("status") and changes["status"] not in VALID_STREAM_ARCHIVE_STATUSES:
            raise ValueError(f"无效录像归档状态：{changes['status']}")
        changes["updated_at"] = self._timestamp()
        assignments = ", ".join(f"{name} = ?" for name in changes)
        parameters = [changes[name] for name in changes] + [source_id]
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE stream_archive_state SET {assignments} WHERE source_id = ?",
                tuple(parameters),
            )
            if cursor.rowcount == 0:
                raise LookupError(f"找不到录像归档状态：{source_id}")
            row = connection.execute(
                "SELECT * FROM stream_archive_state WHERE source_id = ?",
                (source_id,),
            ).fetchone()
        return self._stream_archive_state_from_row(row)

    def get_stream_archive_state(
        self, source_id: str
    ) -> Optional[StreamArchiveStateRecord]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM stream_archive_state WHERE source_id = ?",
                (source_id,),
            ).fetchone()
        return self._stream_archive_state_from_row(row) if row else None

    def list_stream_archive_states(self) -> List[StreamArchiveStateRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM stream_archive_state ORDER BY source_id"
            ).fetchall()
        return [self._stream_archive_state_from_row(row) for row in rows]

    def record_stream_archive_segment(
        self,
        source_id: str,
        *,
        started_at: str | datetime,
        ended_at: str | datetime,
        status: str,
        video_path: str = "",
        duration_seconds: float = 0.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> StreamArchiveSegmentRecord:
        if status not in VALID_STREAM_ARCHIVE_SEGMENT_STATUSES:
            raise ValueError(f"无效录像片段状态：{status}")
        normalized_start = _normalized_utc_time(started_at, "started_at")
        normalized_end = _normalized_utc_time(ended_at, "ended_at")
        if normalized_end <= normalized_start:
            raise ValueError("ended_at 必须晚于 started_at")
        identity = f"{source_id}|{normalized_start}|{normalized_end}"
        segment_id = f"archive-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]}"
        timestamp = self._timestamp()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO stream_archive_segments(
                    segment_id, source_id, video_path, started_at, ended_at,
                    duration_seconds, status, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id, started_at, ended_at) DO UPDATE SET
                    video_path = CASE
                        WHEN excluded.video_path <> '' THEN excluded.video_path
                        ELSE stream_archive_segments.video_path
                    END,
                    duration_seconds = excluded.duration_seconds,
                    status = excluded.status,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    segment_id,
                    source_id,
                    str(video_path),
                    normalized_start,
                    normalized_end,
                    max(0.0, float(duration_seconds)),
                    status,
                    _json_dump(dict(metadata or {})),
                    timestamp,
                    timestamp,
                ),
            )
            row = connection.execute(
                """
                SELECT * FROM stream_archive_segments
                WHERE source_id = ? AND started_at = ? AND ended_at = ?
                """,
                (source_id, normalized_start, normalized_end),
            ).fetchone()
        return self._stream_archive_segment_from_row(row)

    def list_stream_archive_segments(
        self,
        source_id: str,
        *,
        start_time: str | datetime | None = None,
        end_time: str | datetime | None = None,
        statuses: tuple[str, ...] = (),
        limit: int = 10000,
    ) -> List[StreamArchiveSegmentRecord]:
        if limit < 1:
            return []
        unknown = set(statuses) - VALID_STREAM_ARCHIVE_SEGMENT_STATUSES
        if unknown:
            raise ValueError(f"无效录像片段状态：{', '.join(sorted(unknown))}")
        query = "SELECT * FROM stream_archive_segments WHERE source_id = ?"
        parameters: List[Any] = [source_id]
        if start_time is not None:
            normalized_start = _normalized_utc_time(start_time, "start_time")
            query += " AND ended_at > ?"
            parameters.append(normalized_start)
        if end_time is not None:
            normalized_end = _normalized_utc_time(end_time, "end_time")
            query += " AND started_at < ?"
            parameters.append(normalized_end)
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            query += f" AND status IN ({placeholders})"
            parameters.extend(statuses)
        query += " ORDER BY started_at ASC, segment_id ASC LIMIT ?"
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return [self._stream_archive_segment_from_row(row) for row in rows]

    def mark_stream_archive_segment_deleted(
        self, segment_id: str
    ) -> StreamArchiveSegmentRecord:
        timestamp = self._timestamp()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE stream_archive_segments
                SET status = 'deleted', updated_at = ?
                WHERE segment_id = ?
                """,
                (timestamp, segment_id),
            )
            if cursor.rowcount == 0:
                raise LookupError(f"找不到录像片段：{segment_id}")
            row = connection.execute(
                "SELECT * FROM stream_archive_segments WHERE segment_id = ?",
                (segment_id,),
            ).fetchone()
        return self._stream_archive_segment_from_row(row)

    def recover_active_stream_archives(self) -> int:
        timestamp = self._timestamp()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE stream_archive_state
                SET status = 'failed',
                    last_error = '应用进程已重启，录像归档未自动恢复。',
                    updated_at = ?
                WHERE status IN ('starting', 'running', 'stopping')
                """,
                (timestamp,),
            )
            connection.execute(
                """
                UPDATE stream_archive_segments
                SET status = 'failed', updated_at = ?
                WHERE status = 'recording'
                """,
                (timestamp,),
            )
        return int(cursor.rowcount)

    def create_realtime_inspection_task(
        self, session_id: str, *, source_id: str, line_id: str, zone_id: str,
        start_time: str | datetime, end_time: str | datetime, sample_fps: float,
        config: Mapping[str, Any],
    ) -> RealtimeInspectionTaskRecord:
        self.ensure_session(session_id)
        timestamp = self._timestamp()
        task_id = f"realtime-{uuid.uuid4().hex[:12]}"
        start = _normalized_utc_time(start_time, "start_time")
        end = _normalized_utc_time(end_time, "end_time")
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO realtime_inspection_tasks(
                    id, session_id, source_id, line_id, zone_id, start_time, end_time,
                    status, sample_fps, created_at, updated_at, config_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'scheduled', ?, ?, ?, ?)""",
                (task_id, session_id, source_id, line_id, zone_id, start, end,
                 float(sample_fps), timestamp, timestamp, _json_dump(dict(config))),
            )
            row = connection.execute("SELECT * FROM realtime_inspection_tasks WHERE id = ?", (task_id,)).fetchone()
        return self._realtime_task_from_row(row)

    def get_realtime_inspection_task(self, task_id: str) -> Optional[RealtimeInspectionTaskRecord]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM realtime_inspection_tasks WHERE id = ?", (task_id,)).fetchone()
        return self._realtime_task_from_row(row) if row else None

    def list_realtime_inspection_tasks(
        self, *, session_id: Optional[str] = None, source_id: str = "",
        statuses: Optional[tuple[str, ...]] = None, limit: int = 10,
    ) -> List[RealtimeInspectionTaskRecord]:
        clauses: List[str] = []
        values: List[Any] = []
        if session_id is not None: clauses.append("session_id = ?"); values.append(session_id)
        if source_id: clauses.append("source_id = ?"); values.append(source_id)
        if statuses:
            invalid = set(statuses) - VALID_REALTIME_INSPECTION_STATUSES
            if invalid: raise ValueError(f"无效实时巡检状态：{sorted(invalid)}")
            clauses.append("status IN (%s)" % ",".join("?" for _ in statuses)); values.extend(statuses)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM realtime_inspection_tasks{where} ORDER BY created_at DESC LIMIT ?",
                (*values, max(1, min(int(limit), 100))),
            ).fetchall()
        return [self._realtime_task_from_row(row) for row in rows]

    def update_realtime_inspection_task(self, task_id: str, **changes: Any) -> RealtimeInspectionTaskRecord:
        allowed = {"status", "started_at", "stopped_at", "frames_read", "frames_inferred",
                   "frames_dropped", "inference_failures", "reconnect_count", "events_detected",
                   "alarms_created", "highest_risk_level", "last_frame_at", "last_inference_at",
                   "latest_detection_id", "latest_alarm_id", "latest_event_frame",
                   "last_error_code", "last_error_message"}
        unknown = set(changes) - allowed
        if unknown: raise ValueError(f"不支持的实时巡检字段：{sorted(unknown)}")
        if "status" in changes and changes["status"] not in VALID_REALTIME_INSPECTION_STATUSES:
            raise ValueError(f"无效实时巡检状态：{changes['status']}")
        if not changes:
            task = self.get_realtime_inspection_task(task_id)
            if task is None: raise LookupError(f"找不到实时巡检任务：{task_id}")
            return task
        changes["updated_at"] = self._timestamp()
        setters = ", ".join(f"{key} = ?" for key in changes)
        with self._connect() as connection:
            cursor = connection.execute(f"UPDATE realtime_inspection_tasks SET {setters} WHERE id = ?", (*changes.values(), task_id))
            if not cursor.rowcount: raise LookupError(f"找不到实时巡检任务：{task_id}")
            row = connection.execute("SELECT * FROM realtime_inspection_tasks WHERE id = ?", (task_id,)).fetchone()
        return self._realtime_task_from_row(row)

    def record_realtime_inspection_event(
        self, *, event_id: str, task_id: str, source_id: str, detected_at: str,
        ended_at: str, class_name: str, confidence: float, bbox: List[float],
        risk_level: str, detection_id: str, alarm_id: str, image_path: str,
        metadata: Mapping[str, Any], line_id: str = "", last_seen_at: str = "",
        event_status: str = "closed", hit_count: int = 1,
        class_counts: Optional[Mapping[str, int]] = None,
        max_confidence: Optional[float] = None, alarm_report: str = "",
        llm_summary: str = "",
    ) -> RealtimeInspectionEventRecord:
        if event_status not in {"active", "closed"}:
            raise ValueError("event_status 只能是 active 或 closed")
        timestamp = self._timestamp()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO realtime_inspection_events(event_id, task_id, source_id,
                   detected_at, ended_at, class_name, confidence, bbox_json, risk_level,
                   detection_id, alarm_id, image_path, metadata_json, created_at,
                   line_id, last_seen_at, event_status, hit_count, class_counts_json,
                   max_confidence, alarm_report, llm_summary, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (event_id, task_id, source_id, detected_at, ended_at, class_name,
                 float(confidence), _json_dump(list(bbox)), risk_level, detection_id,
                 alarm_id, image_path, _json_dump(dict(metadata)), timestamp,
                 str(line_id or ""), str(last_seen_at or ended_at), event_status,
                 max(1, int(hit_count)), _json_dump(dict(class_counts or {class_name: 1})),
                 float(max_confidence if max_confidence is not None else confidence),
                 str(alarm_report or ""), str(llm_summary or ""), timestamp),
            )
            row = connection.execute("SELECT * FROM realtime_inspection_events WHERE event_id = ?", (event_id,)).fetchone()
        return self._realtime_event_from_row(row)

    def get_realtime_inspection_event(
        self, event_id: str
    ) -> Optional[RealtimeInspectionEventRecord]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM realtime_inspection_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return self._realtime_event_from_row(row) if row else None

    def update_realtime_inspection_event(
        self, event_id: str, **changes: Any
    ) -> RealtimeInspectionEventRecord:
        allowed = {
            "ended_at", "last_seen_at", "event_status", "class_name",
            "confidence", "max_confidence", "bbox", "risk_level", "image_path",
            "hit_count", "class_counts", "metadata", "alarm_report", "llm_summary",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"不支持的实时事件字段：{sorted(unknown)}")
        if changes.get("event_status") not in (None, "active", "closed"):
            raise ValueError("event_status 只能是 active 或 closed")
        column_names = {
            "bbox": "bbox_json", "class_counts": "class_counts_json",
            "metadata": "metadata_json",
        }
        serialized = dict(changes)
        for name in ("bbox", "class_counts", "metadata"):
            if name in serialized:
                serialized[name] = _json_dump(serialized[name])
        serialized["updated_at"] = self._timestamp()
        assignments = ", ".join(
            f"{column_names.get(name, name)} = ?" for name in serialized
        )
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE realtime_inspection_events SET {assignments} WHERE event_id = ?",
                (*serialized.values(), event_id),
            )
            if not cursor.rowcount:
                raise LookupError(f"找不到实时巡检事件：{event_id}")
            row = connection.execute(
                "SELECT * FROM realtime_inspection_events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return self._realtime_event_from_row(row)

    def list_realtime_inspection_events(
        self, task_id: str, limit: Optional[int] = 20, *,
        active_only: bool = False, after_event_id: str = "", latest: bool = False,
    ) -> List[RealtimeInspectionEventRecord]:
        clauses = ["task_id = ?"]
        values: List[Any] = [task_id]
        if active_only:
            clauses.append("event_status = 'active'")
        if after_event_id:
            clauses.append(
                "rowid > COALESCE((SELECT rowid FROM realtime_inspection_events "
                "WHERE event_id = ? AND task_id = ?), 0)"
            )
            values.extend((after_event_id, task_id))
        order = "ASC" if after_event_id else "DESC"
        effective_limit = 1 if latest else limit
        query = (
            "SELECT * FROM realtime_inspection_events WHERE "
            + " AND ".join(clauses)
            + f" ORDER BY rowid {order}"
        )
        with self._connect() as connection:
            if effective_limit is None:
                rows = connection.execute(query, tuple(values)).fetchall()
            else:
                rows = connection.execute(
                    query + " LIMIT ?",
                    (*values, max(1, min(int(effective_limit), 100))),
                ).fetchall()
        return [self._realtime_event_from_row(row) for row in rows]

    def update_detection_alarm(
        self, detection_id: str, alarm_id: str, *, detection: Mapping[str, Any],
        alarm_document: Mapping[str, Any], alarm_report: str,
        source_ended_at: str,
    ) -> tuple[DetectionRecord, AlarmRecord]:
        overall = alarm_document.get("overall_risk") or {}
        risk_level = str(overall.get("level") or "none").lower()
        timestamp = self._timestamp()
        with self._connect() as connection:
            detection_row = connection.execute(
                "SELECT * FROM detection_runs WHERE id = ?", (detection_id,)
            ).fetchone()
            alarm_row = connection.execute(
                "SELECT * FROM alarms WHERE id = ? AND detection_id = ?",
                (alarm_id, detection_id),
            ).fetchone()
            if detection_row is None or alarm_row is None:
                raise LookupError("找不到需要更新的实时检测或报警记录")
            connection.execute(
                """UPDATE detection_runs SET status = ?, risk_level = ?, summary_json = ?,
                   alarm_report = ?, source_ended_at = ? WHERE id = ?""",
                (str(detection.get("status") or "completed"), risk_level,
                 _json_dump(dict(detection)), alarm_report, source_ended_at, detection_id),
            )
            connection.execute(
                """UPDATE alarms SET risk_level = ?, requires_stop = ?, report_json = ?,
                   report_text = ?, updated_at = ? WHERE id = ?""",
                (risk_level, int(bool(overall.get("requires_stop"))),
                 _json_dump(dict(alarm_document)), alarm_report, timestamp, alarm_id),
            )
            detection_row = connection.execute(
                "SELECT * FROM detection_runs WHERE id = ?", (detection_id,)
            ).fetchone()
            alarm_row = connection.execute(
                "SELECT * FROM alarms WHERE id = ?", (alarm_id,)
            ).fetchone()
        return self._detection_from_row(detection_row), self._alarm_from_row(alarm_row)

    def interrupt_active_realtime_inspections(self) -> int:
        timestamp = self._timestamp()
        with self._connect() as connection:
            active_rows = connection.execute(
                "SELECT id FROM realtime_inspection_tasks WHERE status IN "
                "('scheduled','connecting','running','reconnecting','stop_requested')"
            ).fetchall()
            active_ids = [str(row["id"]) for row in active_rows]
            cursor = connection.execute(
                """UPDATE realtime_inspection_tasks SET status = 'interrupted', stopped_at = ?,
                   updated_at = ?, last_error_code = 'task_interrupted',
                   last_error_message = 'Web 服务重启，实时巡检任务已中断，未自动恢复。'
                   WHERE status IN ('scheduled','connecting','running','reconnecting','stop_requested')""",
                (timestamp, timestamp),
            )
            if active_ids:
                placeholders = ",".join("?" for _ in active_ids)
                connection.execute(
                    f"""UPDATE realtime_inspection_events
                    SET event_status = 'closed', ended_at = COALESCE(NULLIF(last_seen_at, ''), ?),
                        updated_at = ?
                    WHERE event_status = 'active' AND task_id IN ({placeholders})""",
                    (timestamp, timestamp, *active_ids),
                )
                connection.execute(
                    f"""UPDATE detection_runs SET source_ended_at = ?
                    WHERE id IN (SELECT detection_id FROM realtime_inspection_events
                    WHERE task_id IN ({placeholders}))""",
                    (timestamp, *active_ids),
                )
            return int(cursor.rowcount)
