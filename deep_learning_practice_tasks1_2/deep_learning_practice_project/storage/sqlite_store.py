from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional


VALID_ALARM_STATUSES = {"inactive", "pending", "confirmed", "cancelled"}
VALID_ALARM_ACTIONS = {"confirm", "cancel"}


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_load(value: str) -> Any:
    return json.loads(value) if value else {}


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

                CREATE INDEX IF NOT EXISTS idx_messages_session_created
                    ON messages(session_id, created_at, id);
                CREATE INDEX IF NOT EXISTS idx_detection_session_created
                    ON detection_runs(session_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_alarm_session_created
                    ON alarms(session_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_alarm_level_created
                    ON alarms(risk_level, created_at);
                """
            )

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
                    risk_level, summary_json, alarm_report, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    def get_alarm(self, alarm_id: str) -> Optional[AlarmRecord]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM alarms WHERE id = ?", (alarm_id,)
            ).fetchone()
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
