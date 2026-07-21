from __future__ import annotations

import threading
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Mapping, Optional

from storage import MonitoringTaskRecord, SQLiteHistoryStore


MonitoringRunner = Callable[[str, Dict[str, Any]], Dict[str, Any]]
TERMINAL_MONITORING_STATUSES = frozenset(
    {"completed", "stopped", "failed", "interrupted"}
)
ACTIVE_MONITORING_STATUSES = ("scheduled", "running", "stop_requested")


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


class MonitoringTaskManager:
    """Run bounded monitoring jobs in daemon threads with SQLite as truth."""

    def __init__(
        self,
        store: SQLiteHistoryStore,
        runner: MonitoringRunner,
        *,
        now: Optional[Callable[[], datetime]] = None,
        recover_orphans: bool = True,
    ) -> None:
        self.store = store
        self._runner = runner
        self._now = now or (lambda: datetime.now().astimezone())
        self._lock = threading.Lock()
        self._stop_events: Dict[str, threading.Event] = {}
        self._threads: Dict[str, threading.Thread] = {}
        if recover_orphans:
            self.store.interrupt_active_monitoring_tasks()

    def start_task(
        self,
        session_id: str,
        *,
        source_id: str,
        line_id: str,
        zone_id: str,
        start_time: str | datetime,
        end_time: str | datetime,
        config: Mapping[str, Any],
    ) -> MonitoringTaskRecord:
        start = _aware_datetime(start_time, "start_time")
        end = _aware_datetime(end_time, "end_time")
        current = self._aware_now()
        if end <= current:
            raise ValueError("end_time 必须晚于当前时间")
        if end <= start:
            raise ValueError("end_time 必须晚于 start_time")
        if (end - start).total_seconds() > 86400:
            raise ValueError("非全天候监控任务最长为 24 小时")
        effective_start = max(start, current)
        task = self.store.create_monitoring_task(
            session_id,
            source_id=source_id,
            line_id=line_id,
            zone_id=zone_id,
            start_time=effective_start.isoformat(timespec="seconds"),
            end_time=end.isoformat(timespec="seconds"),
            config=dict(config),
        )
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._execute_task,
            args=(task.id, stop_event),
            name=f"monitoring-{task.id}",
            daemon=True,
        )
        with self._lock:
            self._stop_events[task.id] = stop_event
            self._threads[task.id] = thread
        thread.start()
        return task

    def stop_task(self, task_id: str, *, session_id: str) -> MonitoringTaskRecord:
        task = self.store.get_monitoring_task(task_id)
        if task is None or task.session_id != session_id:
            raise LookupError(f"找不到当前会话的监控任务：{task_id}")
        if task.status in TERMINAL_MONITORING_STATUSES:
            return task
        with self._lock:
            stop_event = self._stop_events.get(task_id)
        if stop_event is None:
            return self.store.interrupt_monitoring_task_if_active(
                task_id,
                error_code="runner_unavailable",
                error_message="任务运行器已不可用，任务未继续执行。",
            )
        task = self.store.request_monitoring_task_stop(task_id)
        stop_event.set()
        return task

    def wait_for_terminal(
        self,
        task_id: str,
        *,
        timeout_seconds: float,
    ) -> MonitoringTaskRecord:
        with self._lock:
            thread = self._threads.get(task_id)
        if thread is not None:
            thread.join(max(0.0, timeout_seconds))
        task = self.store.get_monitoring_task(task_id)
        if task is None:
            raise LookupError(f"找不到监控任务：{task_id}")
        return task

    def _execute_task(self, task_id: str, stop_event: threading.Event) -> None:
        active_segment_id = ""
        try:
            task = self._required_task(task_id)
            start = _aware_datetime(task.start_time, "start_time")
            end = _aware_datetime(task.end_time, "end_time")
            wait_seconds = max(0.0, (start - self._aware_now()).total_seconds())
            if wait_seconds and stop_event.wait(wait_seconds):
                self.store.update_monitoring_task(task_id, status="stopped")
                return
            if stop_event.is_set():
                self.store.update_monitoring_task(task_id, status="stopped")
                return
            if self._aware_now() >= end:
                self.store.update_monitoring_task(task_id, status="completed")
                return
            self.store.update_monitoring_task(task_id, status="running")

            while not stop_event.is_set():
                task = self._required_task(task_id)
                if self._aware_now() >= end:
                    self.store.update_monitoring_task(task_id, status="completed")
                    return
                run_started_at = self._aware_now()
                run_started = run_started_at.isoformat(timespec="seconds")
                remaining_before_run = (end - run_started_at).total_seconds()
                if remaining_before_run <= 0:
                    self.store.update_monitoring_task(task_id, status="completed")
                    return
                duration_seconds = min(
                    float(task.config["capture_duration_seconds"]),
                    remaining_before_run,
                )
                arguments: Dict[str, Any] = {
                    "source_id": task.source_id,
                    "duration_seconds": duration_seconds,
                    "parameters": dict(task.config.get("parameters") or {}),
                }
                if task.zone_id:
                    arguments["zone_id"] = task.zone_id
                self.store.update_monitoring_job(task_id, status="connecting")
                segment, should_process = self.store.claim_stream_segment(
                    task_id,
                    source_id=task.source_id,
                    started_at=run_started_at,
                    ended_at=min(
                        run_started_at + timedelta(seconds=duration_seconds),
                        end,
                    ),
                )
                active_segment_id = segment.segment_id
                if not should_process:
                    active_segment_id = ""
                    self.store.update_monitoring_job(
                        task_id,
                        status="running",
                        last_processed_at=segment.ended_at,
                    )
                    remaining = (end - self._aware_now()).total_seconds()
                    if remaining <= 0:
                        self.store.update_monitoring_task(task_id, status="completed")
                        return
                    if stop_event.wait(
                        min(float(task.config["interval_seconds"]), remaining)
                    ):
                        self.store.update_monitoring_task(task_id, status="stopped")
                        return
                    continue
                try:
                    result = self._runner(task.session_id, arguments)
                except Exception:
                    result = {
                        "ok": False,
                        "error_code": "monitoring_run_failed",
                        "reply": "本轮监控检测执行失败。",
                        "data": {},
                    }
                run_ended = self._aware_now().isoformat(timespec="seconds")
                data = result.get("data") if isinstance(result, Mapping) else {}
                if not isinstance(data, Mapping):
                    data = {}
                succeeded = bool(result.get("ok"))
                run_summary = self._run_summary(result)
                self.store.finish_stream_segment(
                    active_segment_id,
                    succeeded=succeeded,
                    video_path=str(run_summary.get("video_path") or ""),
                    detection_id=str(data.get("detection_id") or ""),
                )
                active_segment_id = ""
                task, _ = self.store.record_monitoring_run(
                    task_id,
                    succeeded=succeeded,
                    started_at=run_started,
                    ended_at=run_ended,
                    detection_id=str(data.get("detection_id") or ""),
                    alarm_id=str(data.get("alarm_id") or ""),
                    risk_level=str(data.get("risk_level") or ""),
                    error_code=str(result.get("error_code") or ""),
                    error_message=(
                        "" if succeeded else str(result.get("reply") or "检测失败")[:1000]
                    ),
                    result=run_summary,
                )
                if (
                    not succeeded
                    and task.consecutive_failures
                    >= int(task.config["max_consecutive_failures"])
                ):
                    self.store.update_monitoring_task(
                        task_id,
                        status="failed",
                        last_error_code=str(result.get("error_code") or "run_failed"),
                        last_error_message="连续检测失败达到上限，任务已停止。",
                    )
                    return
                if stop_event.is_set():
                    self.store.update_monitoring_task(task_id, status="stopped")
                    return
                self.store.update_monitoring_job(
                    task_id,
                    status="running" if succeeded else "connecting",
                )
                remaining = (end - self._aware_now()).total_seconds()
                if remaining <= 0:
                    self.store.update_monitoring_task(task_id, status="completed")
                    return
                delay = min(float(task.config["interval_seconds"]), remaining)
                if stop_event.wait(delay):
                    self.store.update_monitoring_task(task_id, status="stopped")
                    return
            self.store.update_monitoring_task(task_id, status="stopped")
        except Exception:
            try:
                if active_segment_id:
                    self.store.finish_stream_segment(
                        active_segment_id,
                        succeeded=False,
                    )
                self.store.update_monitoring_task(
                    task_id,
                    status="failed",
                    last_error_code="task_runner_failed",
                    last_error_message="监控任务运行器异常终止。",
                )
            except (LookupError, ValueError):
                pass
        finally:
            with self._lock:
                self._stop_events.pop(task_id, None)
                self._threads.pop(task_id, None)

    def _required_task(self, task_id: str) -> MonitoringTaskRecord:
        task = self.store.get_monitoring_task(task_id)
        if task is None:
            raise LookupError(f"找不到监控任务：{task_id}")
        return task

    def _aware_now(self) -> datetime:
        current = self._now()
        return current if current.tzinfo is not None else current.astimezone()

    @staticmethod
    def _run_summary(result: Mapping[str, Any]) -> Dict[str, Any]:
        data = result.get("data")
        if not isinstance(data, Mapping):
            data = {}
        capture = data.get("capture")
        if not isinstance(capture, Mapping):
            capture = {}
        return {
            "ok": bool(result.get("ok")),
            "reply": str(result.get("reply") or "")[:1000],
            "error_code": str(result.get("error_code") or ""),
            "detection_id": str(data.get("detection_id") or ""),
            "alarm_id": str(data.get("alarm_id") or ""),
            "risk_level": str(data.get("risk_level") or ""),
            "event_count": int(data.get("event_count") or 0),
            "class_counts": dict(data.get("class_counts") or {}),
            "video_path": str(capture.get("video_path") or ""),
        }
