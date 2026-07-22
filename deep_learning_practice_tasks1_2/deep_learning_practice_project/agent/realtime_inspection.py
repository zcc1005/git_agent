from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from storage import RealtimeInspectionTaskRecord, SQLiteHistoryStore

from .streaming import RtspStreamProbe
from .video_sources import LongVideoSource


ACTIVE_STATUSES = ("scheduled", "connecting", "running", "reconnecting", "stop_requested")
TERMINAL_STATUSES = frozenset({"completed", "stopped", "failed", "interrupted"})
RISK_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}


class RealtimeInspectionError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class LatestFrameBuffer:
    """A one-slot buffer which always prefers the newest sampled frame."""

    def __init__(self, maxsize: int = 1) -> None:
        if maxsize not in (1, 2):
            raise ValueError("实时帧缓冲区 maxsize 只能是 1 或 2")
        self.queue: queue.Queue[tuple[Any, datetime, float]] = queue.Queue(maxsize=maxsize)
        self.dropped = 0

    @property
    def maxsize(self) -> int:
        return self.queue.maxsize

    def put_latest(self, item: tuple[Any, datetime, float]) -> None:
        while True:
            try:
                self.queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    self.queue.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass

    def clear(self) -> None:
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                return


class OpenCVRealtimeReader:
    """One long-lived OpenCV RTSP connection; reconnects are explicit."""

    def __init__(self, source: LongVideoSource, *, probe: Optional[RtspStreamProbe] = None) -> None:
        self.source = source
        self._probe = probe or RtspStreamProbe()
        self._capture: Any = None

    def open(self) -> bool:
        if not self.source.is_rtsp or self.source.stream is None:
            return False
        self.release()
        try:
            cv2 = self._probe._load_cv2()
            capture = cv2.VideoCapture()
            url = self.source.resolve_stream_url()
            opened = self._probe._open_capture(capture, cv2, url, self.source)
            if not opened or not capture.isOpened():
                capture.release()
                return False
            self._capture = capture
            return True
        except Exception:
            self.release()
            return False

    def read(self) -> tuple[bool, Any]:
        if self._capture is None:
            return False, None
        try:
            return self._capture.read()
        except Exception:
            return False, None

    def release(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass


class YoloRealtimeFrameDetector:
    """Single-frame adapter over the existing cached model and post-processing."""

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        from project_config import YOLO_MODEL_PATH
        from video_detection import LightweightTracker, _load_model

        self.parameters = dict(parameters)
        self.model = _load_model(str(YOLO_MODEL_PATH.resolve()))
        self.tracker = LightweightTracker(
            known_conf=float(self.parameters.get("known_conf", 0.40)),
            track_max_age_seconds=float(self.parameters.get("track_max_age_seconds", 1.0)),
            min_unknown_hits=int(self.parameters.get("min_unknown_hits", 2)),
            unknown_single_frame_conf=float(self.parameters.get("unknown_single_frame_conf", 0.35)),
            track_iou=float(self.parameters.get("track_iou", 0.15)),
            center_distance_ratio=float(self.parameters.get("track_center_distance_ratio", 0.75)),
        )

    def __call__(self, frame: Any, offset_seconds: float) -> Dict[str, Any]:
        from video_detection import _raw_result_objects, _validated_roi, process_frame_objects

        roi_value = self.parameters.get("roi")
        roi = tuple(int(value) for value in roi_value) if roi_value else None
        effective_roi = _validated_roi(roi, int(frame.shape[1]), int(frame.shape[0]))
        if effective_roi is None:
            inference_frame, offset_x, offset_y = frame, 0, 0
        else:
            x1, y1, x2, y2 = effective_roi
            inference_frame, offset_x, offset_y = frame[y1:y2, x1:x2], x1, y1
        kwargs = {
            "source": inference_frame,
            "conf": float(self.parameters.get("conf", 0.25)),
            "iou": float(self.parameters.get("nms_iou", 0.45)),
            "imgsz": int(self.parameters.get("imgsz", 640)),
            "agnostic_nms": bool(self.parameters.get("agnostic_nms", False)),
            "save": False,
            "verbose": False,
            "batch": 1,
        }
        try:
            import torch
            context = torch.inference_mode()
        except ImportError:
            from contextlib import nullcontext
            context = nullcontext()
        with context:
            result = self.model.predict(**kwargs)[0]
            raw = _raw_result_objects(result, offset_x=offset_x, offset_y=offset_y)
        del result
        processed = process_frame_objects(
            raw, tracker=self.tracker, offset_seconds=offset_seconds,
            duplicate_iou=float(self.parameters.get("duplicate_iou", 0.65)),
            duplicate_containment=float(self.parameters.get("duplicate_containment", 0.85)),
            class_agnostic_duplicates=bool(self.parameters.get("agnostic_nms", False)),
        )
        # LightweightTracker.all_tracks is useful for finite videos but would grow forever here.
        self.tracker.all_tracks = dict(self.tracker.active_tracks)
        return processed


@dataclass
class ActiveEvent:
    sequence: int
    class_name: str
    class_key: str
    first_seen: datetime
    last_seen: datetime
    bbox: List[float]
    confidence: float
    hit_count: int
    representative_frame: Any
    representative_object: Dict[str, Any]
    track_id: Any = None
    confirmed: bool = False
    event_status: str = "active"
    change_type: str = "candidate"
    representative_updated: bool = True
    counted: bool = False
    detection_id: str = ""
    alarm_id: str = ""
    risk_level: str = "none"
    image_path: str = ""


def _bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


class StreamingEventAggregator:
    """Keep bounded candidates and emit confirmed, updated and closed transitions."""

    def __init__(self, *, min_event_hits: int, silence_seconds: float, max_active_events: int = 32) -> None:
        self.min_event_hits = min_event_hits
        self.silence_seconds = silence_seconds
        self.max_active_events = max_active_events
        self._active: List[ActiveEvent] = []
        self._sequence = 0

    @property
    def active_count(self) -> int:
        return len(self._active)

    def update(self, objects: Sequence[Mapping[str, Any]], frame: Any, observed_at: datetime) -> List[ActiveEvent]:
        transitions = self.expire(observed_at)
        matched_sequences: set[int] = set()
        for raw in objects:
            obj = dict(raw)
            class_key = str(obj.get("class") or obj.get("predicted_class") or "unknown")
            class_name = str(obj.get("class_name") or obj.get("predicted_class_name") or class_key)
            bbox = [float(value) for value in (obj.get("bbox_xyxy") or [])[:4]]
            if len(bbox) != 4:
                continue
            track_id = obj.get("track_id")
            candidates = [
                event for event in self._active
                if event.sequence not in matched_sequences and event.class_key == class_key
                and ((track_id is not None and event.track_id == track_id) or _bbox_iou(event.bbox, bbox) >= 0.35)
            ]
            event = max(candidates, key=lambda item: _bbox_iou(item.bbox, bbox)) if candidates else None
            confidence = float(obj.get("confidence") or 0.0)
            if event is None:
                self._sequence += 1
                event = ActiveEvent(
                    self._sequence, class_name, class_key, observed_at, observed_at,
                    bbox, confidence, 1, frame.copy(), obj, track_id,
                )
                self._active.append(event)
            else:
                event.last_seen = observed_at
                event.bbox = bbox
                event.hit_count += 1
                event.representative_updated = False
                if confidence > event.confidence:
                    event.confidence = confidence
                    event.representative_frame = frame.copy()
                    event.representative_object = obj
                    event.representative_updated = True
                if event.track_id is None:
                    event.track_id = track_id
            matched_sequences.add(event.sequence)
            if not event.confirmed and event.hit_count >= self.min_event_hits:
                event.confirmed = True
                event.change_type = "confirmed"
                transitions.append(event)
            elif event.confirmed:
                event.change_type = "updated"
                transitions.append(event)
        while len(self._active) > self.max_active_events:
            evicted = self._active.pop(0)
            if evicted.confirmed:
                evicted.event_status = "closed"
                evicted.change_type = "closed"
                transitions.append(evicted)
        return transitions

    def expire(self, now: datetime, *, flush: bool = False) -> List[ActiveEvent]:
        transitions, retained = [], []
        for event in self._active:
            silent = (now - event.last_seen).total_seconds() > self.silence_seconds
            if flush or silent:
                if event.confirmed:
                    event.event_status = "closed"
                    event.change_type = "closed"
                    transitions.append(event)
            else:
                retained.append(event)
        self._active = retained
        return transitions

    def clear(self) -> None:
        self._active.clear()


EventSink = Callable[[RealtimeInspectionTaskRecord, ActiveEvent], Mapping[str, Any]]
ReaderFactory = Callable[[LongVideoSource], Any]
DetectorFactory = Callable[[Mapping[str, Any]], Callable[[Any, float], Mapping[str, Any]]]


class RealtimeInspectionManager:
    def __init__(
        self, store: SQLiteHistoryStore, *, reader_factory: Optional[ReaderFactory] = None,
        detector_factory: Optional[DetectorFactory] = None, event_sink: Optional[EventSink] = None,
        detection_lock: Optional[threading.RLock] = None,
        now: Optional[Callable[[], datetime]] = None,
        monotonic: Optional[Callable[[], float]] = None, recover_orphans: bool = True,
    ) -> None:
        self.store = store
        self._reader_factory = reader_factory or (lambda source: OpenCVRealtimeReader(source))
        self._detector_factory = detector_factory or YoloRealtimeFrameDetector
        self._event_sink = event_sink or (lambda task, event: {})
        self._detection_lock = detection_lock or threading.RLock()
        self._now = now or (lambda: datetime.now().astimezone())
        self._monotonic = monotonic or time.monotonic
        self._lock = threading.Lock()
        self._threads: Dict[str, threading.Thread] = {}
        self._stop_events: Dict[str, threading.Event] = {}
        if recover_orphans:
            self.store.interrupt_active_realtime_inspections()

    def start_task(
        self, session_id: str, *, source: LongVideoSource, start_time: datetime,
        end_time: datetime, zone_id: str, sample_fps: float, config: Mapping[str, Any],
    ) -> RealtimeInspectionTaskRecord:
        if not 0.2 <= sample_fps <= 10:
            raise RealtimeInspectionError("invalid_schedule", "sample_fps 必须在 0.2 到 10 之间。")
        current = self._aware_now()
        if start_time.tzinfo is None or end_time.tzinfo is None:
            raise RealtimeInspectionError("invalid_schedule", "start_time 和 end_time 必须包含时区。")
        if start_time < current:
            start_time = current
        if end_time <= current or end_time <= start_time:
            raise RealtimeInspectionError("invalid_schedule", "实时巡检结束时间必须晚于当前时间和开始时间。")
        if (end_time - start_time).total_seconds() > 86400:
            raise RealtimeInspectionError("invalid_schedule", "实时巡检最长运行 24 小时。")
        with self._lock:
            active = self.store.list_realtime_inspection_tasks(statuses=ACTIVE_STATUSES, limit=1)
            if active:
                raise RealtimeInspectionError("gpu_busy", f"已有实时巡检任务正在运行：{active[0].id}")
            task = self.store.create_realtime_inspection_task(
                session_id, source_id=source.source_id, line_id=source.line_id, zone_id=zone_id,
                start_time=start_time, end_time=end_time, sample_fps=sample_fps, config=dict(config),
            )
            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._run, args=(task.id, source, stop_event),
                name=f"realtime-inspection-{task.id}", daemon=True,
            )
            self._stop_events[task.id] = stop_event
            self._threads[task.id] = thread
            thread.start()
        return task

    def stop_task(self, task_id: str, *, session_id: str) -> RealtimeInspectionTaskRecord:
        task = self.store.get_realtime_inspection_task(task_id)
        if task is None or task.session_id != session_id:
            raise RealtimeInspectionError("task_not_found", "找不到当前会话的实时巡检任务。")
        if task.status in TERMINAL_STATUSES:
            return task
        with self._lock:
            stop_event = self._stop_events.get(task_id)
        if stop_event is None:
            raise RealtimeInspectionError("task_not_found", "实时巡检执行器不可用。")
        task = self.store.update_realtime_inspection_task(task_id, status="stop_requested")
        stop_event.set()
        return task

    def wait_for_terminal(self, task_id: str, timeout_seconds: float = 5.0) -> RealtimeInspectionTaskRecord:
        with self._lock:
            thread = self._threads.get(task_id)
        if thread is not None:
            thread.join(max(0.0, timeout_seconds))
        task = self.store.get_realtime_inspection_task(task_id)
        if task is None: raise RealtimeInspectionError("task_not_found", "找不到实时巡检任务。")
        return task

    def _run(self, task_id: str, source: LongVideoSource, stop_event: threading.Event) -> None:
        reader: Any = None
        inference_thread: Optional[threading.Thread] = None
        buffer = LatestFrameBuffer(1)
        inference_stop = threading.Event()
        try:
            task = self._required(task_id)
            start = datetime.fromisoformat(task.start_time.replace("Z", "+00:00"))
            end = datetime.fromisoformat(task.end_time.replace("Z", "+00:00"))
            wait = max(0.0, (start - self._aware_now().astimezone(timezone.utc)).total_seconds())
            if stop_event.wait(wait):
                self._finish(task_id, "stopped"); return
            self.store.update_realtime_inspection_task(task_id, status="connecting")
            reader = self._reader_factory(source)
            failures = 0
            while not stop_event.is_set() and not reader.open():
                failures += 1
                if failures >= int(task.config["max_consecutive_failures"]):
                    self._fail(task_id, "consecutive_failures_exceeded", "连续连接失败达到上限。")
                    return
                self.store.update_realtime_inspection_task(task_id, status="reconnecting", reconnect_count=failures,
                                                           last_error_code="connection_failed",
                                                           last_error_message="RTSP 连接失败，正在重连。")
                if stop_event.wait(float(task.config["reconnect_interval_seconds"])):
                    self._finish(task_id, "stopped"); return
            if stop_event.is_set(): self._finish(task_id, "stopped"); return
            started_at = self._aware_now()
            self.store.update_realtime_inspection_task(task_id, status="running", started_at=started_at.isoformat(timespec="seconds"))
            try:
                detector = self._detector_factory(task.config.get("parameters") or {})
            except Exception:
                self._fail(task_id, "model_unavailable", "YOLO 模型当前不可用。")
                return
            aggregator = StreamingEventAggregator(
                min_event_hits=int(task.config["min_event_hits"]),
                silence_seconds=float(task.config["event_silence_seconds"]),
            )
            inference_thread = threading.Thread(
                target=self._infer_loop,
                args=(task_id, detector, aggregator, buffer, inference_stop, stop_event, started_at),
                name=f"realtime-inference-{task_id}", daemon=True,
            )
            inference_thread.start()
            sample_period = 1.0 / task.sample_fps
            next_sample = self._monotonic()
            frames_read = 0
            reconnects = failures
            consecutive = 0
            while not stop_event.is_set() and self._aware_now().astimezone(timezone.utc) < end:
                ok, frame = reader.read()
                if not ok or frame is None:
                    consecutive += 1
                    reader.release()
                    if consecutive >= int(task.config["max_consecutive_failures"]):
                        self._fail(task_id, "consecutive_failures_exceeded", "连续读取失败达到上限。")
                        stop_event.set(); break
                    reconnects += 1
                    self.store.update_realtime_inspection_task(
                        task_id, status="reconnecting", reconnect_count=reconnects,
                        last_error_code="frame_read_failed", last_error_message="视频帧读取失败，正在重连。",
                    )
                    if stop_event.wait(float(task.config["reconnect_interval_seconds"])): break
                    if not reader.open(): continue
                    self.store.update_realtime_inspection_task(task_id, status="running", reconnect_count=reconnects)
                    continue
                consecutive = 0
                frames_read += 1
                observed = self._aware_now()
                now_mono = self._monotonic()
                if now_mono >= next_sample:
                    buffer.put_latest((frame, observed, max(0.0, (observed - started_at).total_seconds())))
                    next_sample = now_mono + sample_period
                if frames_read == 1 or frames_read % 10 == 0:
                    self.store.update_realtime_inspection_task(
                        task_id, frames_read=frames_read, frames_dropped=buffer.dropped,
                        last_frame_at=observed.isoformat(timespec="seconds"),
                    )
            self.store.update_realtime_inspection_task(
                task_id, frames_read=frames_read, frames_dropped=buffer.dropped,
                last_frame_at=self._aware_now().isoformat(timespec="seconds") if frames_read else "",
            )
            inference_stop.set()
            if inference_thread is not None:
                inference_thread.join()
            if self._required(task_id).status == "failed": return
            self._finish(task_id, "stopped" if stop_event.is_set() else "completed")
        except Exception:
            try: self._fail(task_id, "inference_failed", "实时巡检执行异常。")
            except Exception: pass
        finally:
            inference_stop.set()
            if inference_thread is not None and inference_thread.is_alive():
                inference_thread.join(timeout=5)
            if reader is not None: reader.release()
            buffer.clear()
            with self._lock:
                self._threads.pop(task_id, None)
                self._stop_events.pop(task_id, None)

    def _infer_loop(self, task_id: str, detector: Callable[..., Mapping[str, Any]],
                    aggregator: StreamingEventAggregator, buffer: LatestFrameBuffer,
                    inference_stop: threading.Event, stop_event: threading.Event,
        started_at: datetime) -> None:
        inferred = failures = 0

        def persist_transitions(events: Sequence[ActiveEvent]) -> None:
            nonlocal failures
            for changed in events:
                try:
                    self._persist_event(task_id, changed)
                except Exception:
                    failures += 1
                    self.store.update_realtime_inspection_task(
                        task_id, inference_failures=failures,
                        last_error_code="event_persistence_failed",
                        last_error_message="实时事件写入失败，后续命中将继续重试。",
                    )

        while not inference_stop.is_set() or not buffer.queue.empty():
            try:
                frame, observed, offset = buffer.queue.get(timeout=0.05)
            except queue.Empty:
                persist_transitions(aggregator.expire(self._aware_now()))
                continue
            try:
                with self._detection_lock:
                    processed = detector(frame, offset)
                inferred += 1
                objects = processed.get("objects") or []
                persist_transitions(aggregator.update(objects, frame, observed))
                self.store.update_realtime_inspection_task(
                    task_id, frames_inferred=inferred, frames_dropped=buffer.dropped,
                    last_inference_at=observed.isoformat(timespec="seconds"),
                )
            except Exception:
                failures += 1
                self.store.update_realtime_inspection_task(
                    task_id, inference_failures=failures, last_error_code="inference_failed",
                    last_error_message="单帧推理失败。",
                )
        persist_transitions(aggregator.expire(self._aware_now(), flush=True))
        aggregator.clear()

    def _persist_event(self, task_id: str, event: ActiveEvent) -> None:
        task = self._required(task_id)
        result = dict(self._event_sink(task, event) or {})
        event.detection_id = str(result.get("detection_id") or event.detection_id)
        event.alarm_id = str(result.get("alarm_id") or event.alarm_id)
        event.risk_level = str(result.get("risk_level") or event.risk_level)
        event.image_path = str(result.get("image_path") or event.image_path)
        risk = str(result.get("risk_level") or "none")
        highest = task.highest_risk_level
        if RISK_ORDER.get(risk, 0) > RISK_ORDER.get(highest, 0): highest = risk
        first_persistence = not event.counted and bool(event.detection_id)
        if first_persistence:
            event.counted = True
        self.store.update_realtime_inspection_task(
            task_id,
            events_detected=task.events_detected + (1 if first_persistence else 0),
            alarms_created=task.alarms_created + (
                1 if first_persistence and event.alarm_id else 0
            ),
            highest_risk_level=highest, latest_detection_id=str(result.get("detection_id") or ""),
            latest_alarm_id=str(result.get("alarm_id") or ""),
            latest_event_frame=str(result.get("image_path") or ""),
        )

    def _finish(self, task_id: str, status: str) -> None:
        self.store.update_realtime_inspection_task(
            task_id, status=status, stopped_at=self._aware_now().isoformat(timespec="seconds")
        )

    def _fail(self, task_id: str, code: str, message: str) -> None:
        self.store.update_realtime_inspection_task(
            task_id, status="failed", stopped_at=self._aware_now().isoformat(timespec="seconds"),
            last_error_code=code, last_error_message=message,
        )

    def _required(self, task_id: str) -> RealtimeInspectionTaskRecord:
        task = self.store.get_realtime_inspection_task(task_id)
        if task is None: raise RealtimeInspectionError("task_not_found", "找不到实时巡检任务。")
        return task

    def _aware_now(self) -> datetime:
        value = self._now()
        return value if value.tzinfo is not None else value.astimezone()
