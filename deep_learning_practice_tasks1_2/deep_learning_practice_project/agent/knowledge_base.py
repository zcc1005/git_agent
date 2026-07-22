from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol, Sequence


TRUSTED_CODE_SOURCES = (
    "project_config.py",
    "video_detection.py",
    "web_app.py",
    "agent/archive.py",
    "agent/monitoring.py",
    "agent/realtime_inspection.py",
    "agent/streaming.py",
    "agent/video_sources.py",
    "agent/skills/schemas.py",
    "task2_yolo/detect_yolo.py",
    "task3_alarm/alarm_rule_engine.py",
    "task3_alarm/unified_alarm.py",
)

QUERY_EXPANSIONS = {
    "系统能做什么": "异物检测 YOLO 图片 视频 报警 风险报告 智能体 Skill RTSP 历史记录",
    "怎么使用": "启动 Web 服务 常用口令 Skill 检测 报警 查询",
    "怎么用": "启动 Web 服务 常用口令 Skill 检测 报警 查询",
    "实时检测": "detect-video-source 当前 RTSP 定长采集 检测",
    "周期巡检": "start-monitoring-task 定长 MP4 周期 重新连接",
    "持续实时巡检": "start-realtime-inspection 持续连接 sample_fps 代表帧 不缓存完整视频",
    "实时巡检": "start-realtime-inspection 持续连接 sample_fps 代表帧 实时巡检任务",
    "保存视频": "录像归档 control-stream-archive outputs stream_archive MP4 缓存完整视频",
    "视频保存": "录像归档 control-stream-archive outputs stream_archive MP4",
    "历史录像": "detect-archived-video control-stream-archive manifest SQLite 时间范围",
    "代表帧": "key_frame image_path outputs detections_vis realtime_inspections events",
    "历史记录": "agent_history.sqlite3 detection_runs alarms SQLite outputs",
    "检测历史": "agent_history.sqlite3 detection_runs SQLite 检测记录 detection_id",
    "报警记录": "agent_history.sqlite3 alarms SQLite 报警状态 alarm_id",
    "告警记录": "agent_history.sqlite3 alarms SQLite 报警状态 alarm_id",
    "报警": "control-alarm alarm_rule_engine risk_level alarms pending confirmed cancelled",
    "skill": "skills SKILL contract input_schema 参数 工具",
    "rtsp": "video_sources.json MediaMTX FFmpeg probe-video-source capture-video-source",
    "yolo": "YOLOv8 detect_yolo imgsz conf known_conf unknown NMS",
    "sample_fps": "sample_fps 抽帧 每秒检测帧数",
    "known_conf": "known_conf 已知类别 确认阈值",
    "unknown": "unknown 未知异物 低置信度 待确认候选",
    "connecting": "connecting RTSP 连接 任务状态",
    "reconnecting": "reconnecting 断流 重连 reconnect_interval_seconds",
    "gpu_busy": "gpu_busy GPU 推理任务 detection_lock",
    "mediamtx": "MediaMTX RTSP 8554 main-monitor",
    "ffmpeg": "FFmpeg 循环推流 RTSP MediaMTX",
    "web服务": "python web_app.py 127.0.0.1 5000",
}

STOP_TOKENS = {
    "什么", "怎么", "怎样", "如何", "为何", "为什", "么是", "这个", "那个",
    "系统", "项目", "可以", "是否", "一下", "告诉", "请问", "问题", "功能",
}

DEVELOPER_QUESTION_PATTERN = re.compile(
    r"(?:源码|代码(?:在|位置|目录|文件|实现)|哪个函数|调用.*函数|类名|数据库表|字段名|"
    r"接口(?:路径|地址|路由|端点)|API|二次开发|如何开发|怎么开发|实现原理|技术实现|"
    r"修改哪个文件|在哪个文件.{0,12}(?:修改|实现)|完整路径)",
    re.I,
)
DETAILED_SOURCE_PATTERN = re.compile(r"(?:完整路径|具体来源|来源文件|资料来源|引用来源)")

USER_TERM_TRANSLATIONS = {
    "reconnecting": "正在重新连接",
    "connecting": "正在连接",
    "stop_requested": "正在停止",
    "interrupted": "意外中断",
    "scheduled": "等待开始",
    "completed": "已完成",
    "cancelled": "已取消",
    "confirmed": "已确认",
    "pending": "待处理",
    "running": "运行中",
    "active": "处理中",
    "closed": "已结束",
    "stopped": "已停止",
    "failed": "运行失败",
    "gpu_busy": "检测资源正忙",
    "detection_id": "检测记录编号",
    "alarm_id": "报警编号",
    "event_id": "异物事件编号",
    "source_id": "监控源",
    "risk_level": "风险等级",
    "status": "状态",
    "sample_fps": "抽帧频率",
    "known_conf": "已知异物置信度阈值",
    "query": "查询",
    "confirm": "确认报警",
    "cancel": "取消报警",
    "detection_runs": "检测历史",
    "alarms": "报警记录",
    "monitoring_jobs": "巡检任务记录",
    "realtime_inspection_events": "实时异物事件记录",
    "sqlite": "本地历史记录库",
}

USER_SKILL_TRANSLATIONS = {
    "start-realtime-inspection": "启动持续实时巡检",
    "control-realtime-inspection": "查询或停止持续实时巡检",
    "start-monitoring-task": "启动周期巡检",
    "control-monitoring-task": "查询或停止周期巡检",
    "detect-video-source": "检测监控当前画面",
    "probe-video-source": "检查监控是否在线",
    "capture-video-source": "录制监控片段",
    "control-stream-archive": "录像归档",
    "detect-archived-video": "检测历史录像",
    "control-alarm": "报警控制",
    "query-history": "历史查询",
}

SOURCE_LABELS = {
    "readme.md": "项目使用说明",
    "config/readme.md": "监控配置说明",
    "config/video_sources.json": "监控源配置说明",
    "agent/realtime_inspection.py": "持续实时巡检说明",
    "agent/monitoring.py": "周期巡检说明",
    "agent/streaming.py": "录像归档说明",
    "agent/video_sources.py": "监控源说明",
    "task3_alarm/alarm_rule_engine.py": "风险研判说明",
    "task3_alarm/unified_alarm.py": "报警报告说明",
    "task2_yolo/detect_yolo.py": "异物检测说明",
}

SKILL_SOURCE_LABELS = {
    "control-alarm": "报警控制说明",
    "query-history": "历史查询说明",
    "generate-risk-report": "风险报告说明",
    "start-realtime-inspection": "持续实时巡检说明",
    "control-realtime-inspection": "实时巡检控制说明",
    "start-monitoring-task": "周期巡检说明",
    "control-monitoring-task": "周期巡检控制说明",
    "control-stream-archive": "录像归档说明",
    "detect-archived-video": "历史录像检测说明",
    "probe-video-source": "监控连接说明",
    "detect-video-source": "监控检测说明",
}


@dataclass(frozen=True)
class KnowledgeChunk:
    source: str
    heading: str
    text: str
    index: int


@dataclass(frozen=True)
class KnowledgeHit:
    source: str
    heading: str
    excerpt: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "heading": self.heading,
            "excerpt": self.excerpt,
            "score": round(self.score, 4),
        }


class KnowledgeAnswerer(Protocol):
    def classify_request(
        self,
        message: str,
        history: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        ...

    def answer_project_question(
        self,
        question: str,
        evidence: Sequence[Mapping[str, Any]],
        history: Sequence[Mapping[str, Any]],
    ) -> str:
        ...


class ProjectKnowledgeBase:
    """Lazy, repository-only retrieval for public project knowledge."""

    def __init__(self, project_root: Path | str) -> None:
        self.root = Path(project_root).resolve()
        self._chunks: Optional[list[KnowledgeChunk]] = None
        self._chunk_tokens: list[set[str]] = []
        self._document_frequency: Dict[str, int] = {}

    def trusted_sources(self) -> list[str]:
        return [self._relative(path) for path in self._trusted_paths()]

    def search(
        self,
        query: str,
        *,
        limit: int = 6,
        min_score: float = 0.12,
    ) -> list[KnowledgeHit]:
        self._ensure_index()
        assert self._chunks is not None
        expanded = self._expand_query(query)
        query_tokens = self._tokens(expanded)
        if not query_tokens:
            return []
        total_chunks = max(1, len(self._chunks))
        query_weight = sum(self._idf(token, total_chunks) for token in query_tokens)
        normalized_query = self._normalize(query)
        ranked: list[KnowledgeHit] = []
        for chunk, tokens in zip(self._chunks, self._chunk_tokens):
            overlap = query_tokens & tokens
            if not overlap:
                continue
            overlap_weight = sum(self._idf(token, total_chunks) for token in overlap)
            score = overlap_weight / max(query_weight, 1e-6)
            # Exact technical identifiers such as SQLite, RTSP and sample_fps are
            # strong evidence even when surrounded by unmatched colloquial text.
            if any(re.fullmatch(r"[a-z0-9_./-]{2,}", token) for token in overlap):
                score += 0.18
            normalized_text = self._normalize(f"{chunk.heading} {chunk.text}")
            if normalized_query and len(normalized_query) >= 4 and normalized_query in normalized_text:
                score += 0.35
            heading_tokens = self._tokens(chunk.heading)
            if query_tokens & heading_tokens:
                score += 0.08
            if score < min_score:
                continue
            ranked.append(
                KnowledgeHit(
                    source=chunk.source,
                    heading=chunk.heading,
                    excerpt=chunk.text[:1800].strip(),
                    score=score,
                )
            )
        ranked.sort(key=lambda item: (-item.score, item.source, item.heading))
        selected: list[KnowledgeHit] = []
        per_source: Dict[str, int] = {}
        for hit in ranked:
            if per_source.get(hit.source, 0) >= 2:
                continue
            selected.append(hit)
            per_source[hit.source] = per_source.get(hit.source, 0) + 1
            if len(selected) >= max(1, min(int(limit), 10)):
                break
        return selected

    @staticmethod
    def answer_mode(query: str) -> str:
        return "developer" if DEVELOPER_QUESTION_PATTERN.search(query) else "user"

    @staticmethod
    def source_label(source: str, *, detailed: bool = False) -> str:
        normalized = str(source or "").replace("\\", "/")
        if detailed:
            return normalized
        lowered = normalized.lower()
        if lowered in SOURCE_LABELS:
            return SOURCE_LABELS[lowered]
        skill_match = re.match(r"skills/([^/]+)/(?:skill\.md|references/contract\.md)$", lowered)
        if skill_match:
            return SKILL_SOURCE_LABELS.get(skill_match.group(1), "功能使用说明")
        name = Path(normalized).stem.replace("_", " ").strip()
        return f"{name}说明" if name else "项目说明"

    @staticmethod
    def localize_user_text(text: str) -> str:
        value = str(text or "").strip()
        value = re.sub(r"^根据(?:当前)?项目资料[：:，,\s]*", "", value)
        for source, target in USER_SKILL_TRANSLATIONS.items():
            value = re.sub(re.escape(source), target, value, flags=re.I)
        for source, target in sorted(
            USER_TERM_TRANSLATIONS.items(), key=lambda item: -len(item[0])
        ):
            value = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(source)}(?![A-Za-z0-9_])", target, value, flags=re.I)
        value = re.sub(
            r"(?:[A-Za-z]:\\|/)(?=[A-Za-z0-9_.-]*[A-Za-z_.])"
            r"(?:[A-Za-z0-9_.-]+[\\/]){1,}[A-Za-z0-9_.-]+(?:\.[A-Za-z0-9]+)?",
            "相关内部位置",
            value,
        )
        value = re.sub(r"\b(?:[\w.-]+/)+[\w.-]+\.(?:py|md|json|sqlite3)\b", "相关内部资料", value, flags=re.I)
        value = re.sub(r"\b[\w.-]+\.(?:py|md|sqlite3)\b", "相关内部资料", value, flags=re.I)
        value = re.sub(r"/api/[A-Za-z0-9_./-]+", "网页端对应功能", value, flags=re.I)
        value = re.sub(r"\b[a-z][a-z0-9_]*(?:_[a-z0-9_]+)+\b", "相关参数", value, flags=re.I)
        value = re.sub(r"\b[a-zA-Z_][a-zA-Z0-9_]*\(\)", "内部处理程序", value)
        value = re.sub(
            r"\b[A-Z][A-Za-z0-9]*(?:Manager|Service|Store|Record|Client|Router|Planner)\b",
            "内部组件",
            value,
        )
        value = re.sub(r"[ \t]{2,}", " ", value)
        return value.strip()

    def present_user_answer(self, query: str, answer: str) -> str:
        clean = self.localize_user_text(answer)
        clean = re.sub(r"\n*参考[：:].*$", "", clean, flags=re.S).strip()
        headings = {
            "steps": re.search(r"(?:^|\n)操作步骤[：:]?", clean),
            "note": re.search(r"(?:^|\n)注意事项[：:]?", clean),
        }
        starts = [match.start() for match in headings.values() if match]
        direct = clean[: min(starts)].strip() if starts else clean
        direct = re.sub(r"^直接回答[：:]\s*", "", direct).strip()

        def section(name: str) -> str:
            match = headings[name]
            if not match:
                return ""
            later = [
                other.start()
                for key, other in headings.items()
                if key != name and other and other.start() > match.start()
            ]
            end = min(later) if later else len(clean)
            return clean[match.end() : end].strip()

        steps = section("steps")
        note = section("note") or self._attention_note(query)
        parts = [direct or "这个问题可以通过系统现有功能处理。"]
        parts.append(
            f"操作步骤：\n{steps}" if steps else self._operation_steps(query)
        )
        if note:
            parts.append(f"注意事项：{note}")
        return "\n\n".join(parts).strip()

    def fallback_answer(
        self,
        query: str,
        hits: Sequence[KnowledgeHit],
        *,
        answer_mode: str = "user",
    ) -> str:
        if not hits:
            return (
                "现有资料没有说明这个问题，我暂时不能可靠回答。\n\n"
                "操作步骤：请补充你想了解的功能、监控对象或具体操作。"
            )
        if answer_mode == "user":
            return self._user_fallback_answer(query, hits)
        query_tokens = self._tokens(self._expand_query(query))
        candidates: list[tuple[float, str]] = []
        for hit in hits:
            for sentence in self._sentences(hit.excerpt):
                sentence_tokens = self._tokens(sentence)
                overlap = query_tokens & sentence_tokens
                if not overlap or len(sentence.strip()) < 8:
                    continue
                score = len(overlap) / max(1, len(query_tokens)) + hit.score * 0.25
                candidates.append((score, sentence.strip()))
        candidates.sort(key=lambda item: -item[0])
        selected: list[str] = []
        total = 0
        for _, sentence in candidates:
            if sentence in selected:
                continue
            if total + len(sentence) > 650:
                continue
            selected.append(sentence)
            total += len(sentence)
            if len(selected) >= 5:
                break
        if not selected:
            selected = [hits[0].excerpt[:500].strip()]
        return "\n".join(f"- {item}" for item in selected)

    def _user_fallback_answer(
        self, query: str, hits: Sequence[KnowledgeHit]
    ) -> str:
        normalized = query.lower()
        if ("报警" in normalized or "告警" in normalized) and "历史" in normalized:
            direct = (
                "可以直接在聊天框查询历史报警。系统会返回报警时间、异物类别、"
                "风险等级、代表帧和处理状态。"
            )
        elif "抽帧" in normalized or "sample_fps" in normalized:
            direct = (
                "抽帧频率表示每秒最多选取多少帧进行异物检测。数值越高，检查越密集，"
                "但会占用更多检测资源。"
            )
        elif "实时巡检" in normalized or "持续巡检" in normalized:
            direct = (
                "持续实时巡检会保持监控连接，并按设定频率持续检查画面。"
                "默认只保存确认异物事件的代表帧，不保存完整巡检视频。"
            )
        elif "rtsp" in normalized or "监控源" in normalized:
            direct = (
                "接入网络监控前，需要先登记监控源并确认视频流可访问。登记完成后，"
                "即可通过聊天框检查在线状态、录制片段或启动检测。"
            )
        elif any(item in normalized for item in ("connecting", "reconnecting", "gpu_busy")):
            direct = (
                "这些提示分别表示正在连接、正在重新连接，或检测资源正忙。"
                "它们是运行状态提示，不代表检测结论。"
            )
        elif "系统" in normalized and any(item in normalized for item in ("功能", "能做", "支持")):
            direct = (
                "系统支持图片和视频异物检测、监控巡检、风险研判、报警处置、"
                "历史查询和风险报告生成。"
            )
        else:
            query_tokens = self._tokens(self._expand_query(query))
            sentence = ""
            for hit in hits:
                for candidate in self._sentences(hit.excerpt):
                    if query_tokens & self._tokens(candidate):
                        sentence = self.localize_user_text(candidate)
                        if len(sentence) >= 8:
                            break
                if sentence:
                    break
            direct = sentence or "这个功能已有项目资料说明，但当前无法生成更详细的可靠解读。"
        return self.present_user_answer(query, direct)

    @staticmethod
    def _operation_steps(query: str) -> str:
        normalized = query.lower()
        if ("报警" in normalized or "告警" in normalized) and "历史" in normalized:
            return (
                "操作步骤：\n"
                "1. 输入“查看今天的报警记录”。\n"
                "2. 也可以输入“查看主监控最近一次高风险报警”。\n"
                "3. 如需详情，输入“查看某个报警的详细报告”。"
            )
        if "实时巡检" in normalized or "持续巡检" in normalized:
            return (
                "操作步骤：\n"
                "1. 在聊天框说明监控名称、开始时间和结束时间。\n"
                "2. 需要查询时输入“查看实时巡检状态”。\n"
                "3. 需要结束时输入“停止实时巡检”。"
            )
        if "rtsp" in normalized or "监控源" in normalized:
            return (
                "操作步骤：\n"
                "1. 先完成监控源登记和视频流启动。\n"
                "2. 输入“主监控在线吗”检查连接。\n"
                "3. 在线后再输入录制或检测指令。"
            )
        if "抽帧" in normalized or "sample_fps" in normalized:
            return (
                "操作步骤：创建检测或巡检任务时说明每秒需要检测多少帧；"
                "没有特殊要求时可先使用系统默认值。"
            )
        if "报警" in normalized or "告警" in normalized:
            return (
                "操作步骤：在聊天框输入“查看当前报警”；确认现场情况后，再选择确认报警或取消报警。"
            )
        return "操作步骤：在聊天框直接说明要查看的功能、监控对象和时间范围即可。"

    @staticmethod
    def _attention_note(query: str) -> str:
        normalized = query.lower()
        if "实时巡检" in normalized or "持续巡检" in normalized:
            return "持续实时巡检必须设置结束时间或运行时长；完整录像需要单独开启录像归档。"
        if "rtsp" in normalized:
            return "监控地址和登录信息只应保存在受控配置中，不要直接发送到聊天内容。"
        if "报警" in normalized or "告警" in normalized:
            return "报警状态和风险等级以系统实际保存的检测结果为准。"
        return ""

    def _ensure_index(self) -> None:
        if self._chunks is not None:
            return
        chunks: list[KnowledgeChunk] = []
        for path in self._trusted_paths():
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            source = self._relative(path)
            chunks.extend(self._split_document(source, text, path.suffix.lower()))
        self._chunks = chunks
        self._chunk_tokens = [self._tokens(f"{item.heading} {item.text}") for item in chunks]
        frequency: Dict[str, int] = {}
        for tokens in self._chunk_tokens:
            for token in tokens:
                frequency[token] = frequency.get(token, 0) + 1
        self._document_frequency = frequency

    def _trusted_paths(self) -> list[Path]:
        paths: list[Path] = []
        direct = [
            self.root / "README.md",
            self.root / "config" / "README.md",
            self.root / "config" / "video_sources.json",
            *(self.root / name for name in TRUSTED_CODE_SOURCES),
        ]
        paths.extend(path for path in direct if path.is_file())
        docs = self.root / "docs"
        if docs.is_dir():
            paths.extend(path for path in docs.rglob("*.md") if path.is_file())
        skills = self.root / "skills"
        if skills.is_dir():
            paths.extend(path for path in skills.glob("*/SKILL.md") if path.is_file())
            paths.extend(
                path for path in skills.glob("*/references/contract.md") if path.is_file()
            )
        unique: Dict[str, Path] = {}
        for path in paths:
            resolved = path.resolve()
            try:
                resolved.relative_to(self.root)
            except ValueError:
                continue
            unique[str(resolved).lower()] = resolved
        return sorted(unique.values(), key=lambda item: self._relative(item).lower())

    def _split_document(
        self, source: str, text: str, suffix: str
    ) -> list[KnowledgeChunk]:
        if suffix == ".md":
            return self._split_markdown(source, text)
        if suffix == ".json":
            try:
                text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                pass
        return self._split_blocks(source, text, heading=source)

    def _split_markdown(self, source: str, text: str) -> list[KnowledgeChunk]:
        sections: list[tuple[str, str]] = []
        heading = source
        body: list[str] = []
        for line in text.splitlines():
            match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
            if match:
                if any(item.strip() for item in body):
                    sections.append((heading, "\n".join(body).strip()))
                heading = match.group(1).strip()
                body = []
            else:
                body.append(line)
        if any(item.strip() for item in body):
            sections.append((heading, "\n".join(body).strip()))
        chunks: list[KnowledgeChunk] = []
        for section_heading, section_text in sections:
            chunks.extend(self._split_blocks(source, section_text, heading=section_heading))
        return chunks

    @staticmethod
    def _split_blocks(
        source: str,
        text: str,
        *,
        heading: str,
        max_chars: int = 1600,
    ) -> list[KnowledgeChunk]:
        paragraphs = re.split(r"\n\s*\n", text)
        chunks: list[KnowledgeChunk] = []
        current: list[str] = []
        size = 0
        index = 0
        for paragraph in paragraphs:
            clean = paragraph.strip()
            if not clean:
                continue
            if current and size + len(clean) + 2 > max_chars:
                chunks.append(KnowledgeChunk(source, heading, "\n\n".join(current), index))
                index += 1
                current = []
                size = 0
            if len(clean) > max_chars:
                for offset in range(0, len(clean), max_chars):
                    piece = clean[offset : offset + max_chars]
                    if current:
                        chunks.append(KnowledgeChunk(source, heading, "\n\n".join(current), index))
                        index += 1
                        current = []
                        size = 0
                    chunks.append(KnowledgeChunk(source, heading, piece, index))
                    index += 1
                continue
            current.append(clean)
            size += len(clean) + 2
        if current:
            chunks.append(KnowledgeChunk(source, heading, "\n\n".join(current), index))
        return chunks

    def _expand_query(self, query: str) -> str:
        normalized = query.lower()
        additions = [value for key, value in QUERY_EXPANSIONS.items() if key in normalized]
        return " ".join([query, *additions])

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", "", text).lower()

    @staticmethod
    def _tokens(text: str) -> set[str]:
        normalized = text.lower()
        tokens = set(re.findall(r"[a-z0-9_./-]{2,}", normalized))
        for block in re.findall(r"[\u4e00-\u9fff]+", normalized):
            for width in (2, 3):
                tokens.update(
                    block[index : index + width]
                    for index in range(max(0, len(block) - width + 1))
                )
        return {token for token in tokens if token not in STOP_TOKENS}

    def _idf(self, token: str, total_chunks: int) -> float:
        frequency = self._document_frequency.get(token, 0)
        return math.log((total_chunks + 1) / (frequency + 1)) + 1.0

    @staticmethod
    def _sentences(text: str) -> Iterable[str]:
        for line in text.splitlines():
            clean = re.sub(r"^[\s>*+-]+", "", line).strip()
            if not clean or clean.startswith("```"):
                continue
            for sentence in re.split(r"(?<=[。！？；])\s*", clean):
                if sentence.strip():
                    yield sentence.strip()

    def _relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()
