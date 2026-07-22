function createMessage(role, text, isError = false, attachment = null) {
  const article = document.createElement("article");
  article.className = `agent-chat__message agent-chat__message--${role}`;
  article.setAttribute("aria-label", role === "user" ? "你的消息" : "智能体消息");
  if (isError) article.classList.add("agent-chat__message--error");
  if (attachment?.src) article.classList.add("agent-chat__message--media");

  if (text) {
    const normalizedText = String(text);
    const referenceMatch = normalizedText.match(/(?:^|\n\n)(参考：[^\n]+)\s*$/);
    const bodyText = referenceMatch
      ? normalizedText.slice(0, referenceMatch.index).trim()
      : normalizedText;
    const content = document.createElement("p");
    content.textContent = bodyText;
    article.append(content);
    if (referenceMatch) {
      const reference = document.createElement("small");
      reference.className = "agent-chat__knowledge-reference";
      reference.textContent = referenceMatch[1];
      article.append(reference);
    }
  }
  if (attachment?.src) {
    const preview = document.createElement("div");
    preview.className = "agent-chat__media-preview";
    if (attachment.media_type === "video") {
      const video = document.createElement("video");
      video.src = attachment.src;
      if (attachment.poster) video.poster = attachment.poster;
      video.controls = true;
      video.preload = "auto";
      video.playsInline = true;
      video.setAttribute("aria-label", "已上传的原视频");
      preview.append(video);
    } else {
      const image = document.createElement("img");
      image.src = attachment.src;
      image.alt = "已上传的原图片";
      preview.append(image);
    }
    article.append(preview);
  }
  return article;
}

function outputPathToUrl(path) {
  const normalized = String(path || "").replaceAll("\\", "/");
  const marker = "/outputs/";
  const markerIndex = normalized.toLowerCase().lastIndexOf(marker);
  const relative = markerIndex >= 0
    ? normalized.slice(markerIndex + marker.length)
    : normalized.startsWith("outputs/")
      ? normalized.slice("outputs/".length)
      : normalized;
  return `/outputs/${relative.split("/").map(encodeURIComponent).join("/")}`;
}

function storedAttachmentPreview(attachment) {
  if (!attachment?.path) return null;
  return {
    media_type: attachment.media_type,
    src: outputPathToUrl(attachment.preview_path || attachment.path),
    poster: attachment.poster_path
      ? outputPathToUrl(attachment.poster_path)
      : "",
  };
}

function extractAlarmReport(data) {
  if (!data || typeof data !== "object") return null;
  if (typeof data.alarm_report === "string" && data.alarm_report.trim()) {
    return {
      text: data.alarm_report.trim(),
      jsonPath: typeof data.alarm_json === "string" ? data.alarm_json : "",
      eventFrames: Array.isArray(data.event_frames) ? data.event_frames : [],
    };
  }
  if (typeof data.report_text === "string" && data.report_text.trim()) {
    return {
      text: data.report_text.trim(),
      jsonPath: "",
      eventFrames: Array.isArray(data.event_frames) ? data.event_frames : [],
    };
  }
  if (data.alarm && typeof data.alarm === "object") {
    const nestedAlarm = extractAlarmReport(data.alarm);
    if (nestedAlarm) return nestedAlarm;
  }
  if (Array.isArray(data.steps)) {
    for (let index = data.steps.length - 1; index >= 0; index -= 1) {
      const stepReport = extractAlarmReport(data.steps[index]?.data);
      if (stepReport) return stepReport;
    }
  }
  return null;
}

function extractDetectionPresentation(data) {
  if (!data || typeof data !== "object") return null;
  if (data.structured_alert?.detection_id) {
    return {
      alert: data.structured_alert,
      analysis: String(data.ai_analysis || "").trim(),
      analysisSource: String(data.analysis_source || ""),
      quickQuestions: Array.isArray(data.quick_questions) ? data.quick_questions : [],
    };
  }
  if (Array.isArray(data.steps)) {
    for (let index = data.steps.length - 1; index >= 0; index -= 1) {
      const nested = extractDetectionPresentation(data.steps[index]?.data);
      if (nested) return nested;
    }
  }
  if (Array.isArray(data.segment_results)) {
    for (let index = data.segment_results.length - 1; index >= 0; index -= 1) {
      const nested = extractDetectionPresentation(data.segment_results[index]?.data);
      if (nested) return nested;
    }
  }
  if (data.data && typeof data.data === "object") {
    return extractDetectionPresentation(data.data);
  }
  return null;
}

function extractRealtimeReport(data) {
  if (!data || typeof data !== "object") return null;
  if (data.realtime_report?.task_id) return data.realtime_report;
  if (Array.isArray(data.steps)) {
    for (let index = data.steps.length - 1; index >= 0; index -= 1) {
      const nested = extractRealtimeReport(data.steps[index]?.data);
      if (nested) return nested;
    }
  }
  if (data.data && typeof data.data === "object") {
    return extractRealtimeReport(data.data);
  }
  return null;
}

function findDetectionId(data) {
  const presentation = extractDetectionPresentation(data);
  if (presentation?.alert?.detection_id) return String(presentation.alert.detection_id);
  if (!data || typeof data !== "object") return "";
  if (data.detection_id) return String(data.detection_id);
  for (const value of Object.values(data)) {
    if (!value || typeof value !== "object") continue;
    const nested = findDetectionId(value);
    if (nested) return nested;
  }
  return "";
}

function displayAlarmReportText(text) {
  let report = String(text || "").trim();
  const conclusionIndex = report.indexOf("二、报警结论");
  if (conclusionIndex >= 0) report = report.slice(conclusionIndex);
  const hiddenSections = ["\n五、风险说明", "\n六、处理建议", "\n七、生成信息"]
    .map((heading) => report.indexOf(heading))
    .filter((index) => index >= 0);
  if (hiddenSections.length) report = report.slice(0, Math.min(...hiddenSections));
  const sectionNumbers = [
    ["二、报警结论", "一、报警结论"],
    ["三、总体风险等级", "二、总体风险等级"],
    ["四、事件详情", "三、事件详情"],
  ];
  sectionNumbers.forEach(([original, displayed]) => {
    report = report.replace(original, displayed);
  });
  return report.trim();
}

function appendReportText(container, text) {
  const normalized = String(text || "").trim();
  if (!normalized) return;
  const content = document.createElement("pre");
  content.textContent = normalized;
  container.append(content);
}

function appendReportContent(container, report) {
  const text = displayAlarmReportText(report.text);
  const eventFrames = Array.isArray(report.eventFrames) ? report.eventFrames : [];
  const frameByEvent = new Map(
    eventFrames.map((item) => [Number(item.event_id), String(item.key_frame || "")]),
  );
  const detailStart = text.indexOf("三、事件详情");
  if (frameByEvent.size === 0 || detailStart < 0) {
    appendReportText(container, text);
    return;
  }

  appendReportText(container, text.slice(0, detailStart));
  const detailText = text.slice(detailStart);
  detailText.split(/(?=^事件\d+：)/gm).forEach((block) => {
    appendReportText(container, block);
    const eventMatch = block.match(/^事件(\d+)：/m);
    const eventId = eventMatch ? Number(eventMatch[1]) : 0;
    const keyFrame = frameByEvent.get(eventId);
    if (!keyFrame) return;
    const figure = document.createElement("figure");
    figure.className = "agent-chat__event-frame";
    const image = document.createElement("img");
    image.src = outputPathToUrl(keyFrame);
    image.alt = `事件${eventId}代表检测帧`;
    image.loading = "lazy";
    image.addEventListener("error", () => figure.remove(), { once: true });
    const caption = document.createElement("figcaption");
    caption.textContent = `事件${eventId}代表检测帧`;
    figure.append(image, caption);
    container.append(figure);
  });
}

function appendDetectionPresentation(article, presentation) {
  if (!article || !presentation?.alert?.detection_id) return;
  const alert = presentation.alert;
  article.classList.add("agent-chat__message--with-report");
  const panel = document.createElement("section");
  panel.className = "agent-chat__structured-alert";
  const title = document.createElement("h3");
  title.textContent = "结构化预警结果";
  const grid = document.createElement("dl");
  const classes = Object.entries(alert.class_counts || {})
    .map(([name, count]) => `${name}${count}个`).join("、") || "无确认异物";
  const confidence = alert.max_confidence == null
    ? "—"
    : Number(alert.max_confidence).toFixed(4);
  const fields = [
    ["监控源", alert.monitor_source || "未知"],
    ["时间", alert.detected_at || "未知"],
    ["异物类别", classes],
    ["数量", String(alert.object_count ?? 0)],
    ["最高置信度", confidence],
    ["风险等级", alert.risk_level_name || alert.risk_level || "未知"],
    ["检测编号", alert.detection_id],
    ["报警状态", alert.alarm_status_name || alert.alarm_status || "未知"],
  ];
  fields.forEach(([label, value]) => {
    const term = document.createElement("dt");
    term.textContent = label;
    const description = document.createElement("dd");
    description.textContent = String(value);
    grid.append(term, description);
  });
  panel.append(title, grid);

  const frames = Array.isArray(alert.representative_frames)
    ? alert.representative_frames
    : [];
  frames.forEach((item, index) => {
    const keyFrame = String(item?.key_frame || item?.image_path || "");
    if (!keyFrame) return;
    const eventId = Number(item?.event_id || index + 1);
    const figure = document.createElement("figure");
    figure.className = "agent-chat__event-frame";
    const image = document.createElement("img");
    image.src = outputPathToUrl(keyFrame);
    image.alt = `事件${eventId}代表检测帧`;
    image.loading = "lazy";
    image.addEventListener("error", () => figure.remove(), { once: true });
    const caption = document.createElement("figcaption");
    caption.textContent = `事件${eventId}代表检测帧`;
    figure.append(image, caption);
    panel.append(figure);
  });

  if (presentation.analysis) {
    const analysis = document.createElement("section");
    analysis.className = "agent-chat__ai-analysis";
    const heading = document.createElement("h3");
    heading.textContent = "AI 智能简析";
    const text = document.createElement("p");
    text.textContent = presentation.analysis;
    analysis.append(heading, text);
    panel.append(analysis);
  }
  article.append(panel);
}

function appendRealtimeReport(article, report) {
  if (!article || !report?.task_id) return;
  article.classList.add("agent-chat__message--with-report");
  const panel = document.createElement("section");
  panel.className = "agent-chat__structured-alert agent-chat__realtime-report";
  const title = document.createElement("h3");
  title.textContent = "实时巡检预警结果";
  const grid = document.createElement("dl");
  const classes = Object.entries(report.class_counts || {})
    .map(([name, count]) => `${name}${count}个`).join("、") || "无确认异物";
  const confidence = report.max_confidence == null
    ? "—"
    : Number(report.max_confidence).toFixed(4);
  const fields = [
    ["监控源", report.monitor_source || "未知"],
    ["巡检时间", `${report.start_time || "未知"} 至 ${report.end_time || "未知"}`],
    ["异物类别", classes],
    ["事件数量", String(report.event_count ?? 0)],
    ["报警数量", String(report.alarm_count ?? 0)],
    ["最高置信度", confidence],
    ["总体风险", report.risk_level_name || report.risk_level || "未知"],
  ];
  fields.forEach(([label, value]) => {
    const term = document.createElement("dt");
    term.textContent = label;
    const description = document.createElement("dd");
    description.textContent = String(value);
    grid.append(term, description);
  });
  panel.append(title, grid);

  const events = Array.isArray(report.events) ? report.events : [];
  if (!events.length) {
    const empty = document.createElement("p");
    empty.className = "agent-chat__realtime-empty";
    empty.textContent = "本轮实时巡检未发现确认异物，因此没有触发报警事件。";
    panel.append(empty);
  }
  events.forEach((event, index) => {
    const section = document.createElement("section");
    section.className = "agent-chat__realtime-event";
    const heading = document.createElement("h4");
    heading.textContent = `事件${event.event_number || index + 1}`;
    const details = document.createElement("dl");
    const eventConfidence = event.confidence == null
      ? "—"
      : Number(event.confidence).toFixed(4);
    [
      ["出现时间", event.detected_at || "未知"],
      ["异物类别", event.class_name || "未知异物"],
      ["置信度", eventConfidence],
      ["目标位置", event.position || "未知区域"],
      ["风险等级", event.risk_level_name || event.risk_level || "未知"],
      ["检测编号", event.detection_id || "—"],
      ["报警状态", event.alarm_status_name || event.alarm_status || "未知"],
    ].forEach(([label, value]) => {
      const term = document.createElement("dt");
      term.textContent = label;
      const description = document.createElement("dd");
      description.textContent = String(value);
      details.append(term, description);
    });
    section.append(heading, details);
    const imagePath = String(event.image_path || "");
    if (imagePath) {
      const figure = document.createElement("figure");
      figure.className = "agent-chat__event-frame";
      const image = document.createElement("img");
      image.src = outputPathToUrl(imagePath);
      image.alt = `事件${event.event_number || index + 1}代表检测帧`;
      image.loading = "lazy";
      image.addEventListener("error", () => figure.remove(), { once: true });
      const caption = document.createElement("figcaption");
      caption.textContent = `事件${event.event_number || index + 1}代表检测帧`;
      figure.append(image, caption);
      section.append(figure);
    }
    panel.append(section);
  });

  const alarmStatuses = report.alarm_status_counts || {};
  const pendingCount = Number(alarmStatuses.pending || 0);
  const closure = document.createElement("section");
  closure.className = "agent-chat__alarm-closure";
  const closureText = document.createElement("p");
  if (pendingCount > 0) {
    closureText.textContent = `本轮还有${pendingCount}条报警待确认，是否确认报警？`;
    const actions = document.createElement("div");
    actions.className = "agent-chat__alarm-actions";
    [
      ["confirm", "确认本轮报警"],
      ["cancel", "取消本轮报警"],
    ].forEach(([action, label]) => {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.agentAlarmAction = action;
      button.dataset.taskId = String(report.task_id);
      button.textContent = label;
      actions.append(button);
    });
    closure.append(closureText, actions);
  } else if (Number(alarmStatuses.confirmed || 0) > 0) {
    closureText.textContent = `本轮报警已闭环：已确认${Number(alarmStatuses.confirmed)}条。`;
    closure.append(closureText);
  } else if (Number(alarmStatuses.cancelled || 0) > 0) {
    closureText.textContent = `本轮报警已闭环：已取消${Number(alarmStatuses.cancelled)}条。`;
    closure.append(closureText);
  }
  if (closure.childNodes.length) panel.append(closure);

  if (report.ai_analysis) {
    const analysis = document.createElement("section");
    analysis.className = "agent-chat__ai-analysis";
    const heading = document.createElement("h3");
    heading.textContent = "AI 智能简析";
    const text = document.createElement("p");
    text.textContent = report.ai_analysis;
    analysis.append(heading, text);
    panel.append(analysis);
  }
  article.append(panel);
}

function realtimeEventKey(event) {
  return `${String(event?.task_id || "")}::${String(event?.event_id || "")}`;
}

function appendRealtimeEvent(article, event) {
  if (!article || !event?.event_id || !event?.task_id) return;
  article.classList.add("agent-chat__message--with-report");
  article.dataset.realtimeEventKey = realtimeEventKey(event);
  const panel = document.createElement("section");
  panel.className = "agent-chat__structured-alert agent-chat__realtime-live-event";
  const title = document.createElement("h3");
  title.textContent = "【实时异物预警】";
  const grid = document.createElement("dl");
  const classCounts = Object.entries(event.class_counts || {});
  const quantity = classCounts.reduce((sum, [, count]) => sum + Number(count || 0), 0) || 1;
  const confidence = Number(event.max_confidence ?? event.confidence);
  [
    ["发现时间", event.detected_at || "未知"],
    ["异物类别", event.class_name || "未知异物"],
    ["数量", String(quantity)],
    ["最高置信度", Number.isFinite(confidence) ? confidence.toFixed(4) : "—"],
    ["风险等级", event.risk_level_name || event.risk_level || "未知"],
    ["报警状态", event.alarm_status_name || event.alarm_status || "未知"],
    ["事件状态", event.event_status === "closed" ? "已关闭" : "持续中"],
  ].forEach(([label, value]) => {
    const term = document.createElement("dt");
    term.textContent = label;
    const description = document.createElement("dd");
    description.textContent = String(value);
    if (label === "事件状态") description.dataset.realtimeEventStatus = "";
    grid.append(term, description);
  });
  panel.append(title, grid);
  const imagePath = String(event.representative_frame || event.image_path || "");
  if (imagePath) {
    const figure = document.createElement("figure");
    figure.className = "agent-chat__event-frame";
    const image = document.createElement("img");
    image.src = outputPathToUrl(imagePath);
    image.alt = "实时异物事件代表帧";
    image.loading = "lazy";
    image.addEventListener("error", () => figure.remove(), { once: true });
    const caption = document.createElement("figcaption");
    caption.textContent = "代表帧";
    figure.append(image, caption);
    panel.append(figure);
  }
  if (event.llm_summary) {
    const analysis = document.createElement("section");
    analysis.className = "agent-chat__ai-analysis";
    const heading = document.createElement("h3");
    heading.textContent = "AI 智能简析";
    const text = document.createElement("p");
    text.textContent = String(event.llm_summary);
    analysis.append(heading, text);
    panel.append(analysis);
  }
  article.append(panel);
}

function newSessionId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

const TERMINAL_MONITORING_STATUSES = new Set(["completed", "failed", "cancelled"]);
const TERMINAL_REALTIME_STATUSES = new Set(["completed", "stopped", "failed", "interrupted"]);

function findMonitoringTaskId(value) {
  if (!value || typeof value !== "object") return "";
  if (value.monitoring_job?.task_id) return String(value.monitoring_job.task_id);
  if (Array.isArray(value.steps)) {
    for (let index = value.steps.length - 1; index >= 0; index -= 1) {
      const nested = findMonitoringTaskId(value.steps[index]?.data);
      if (nested) return nested;
    }
  }
  if (value.data && typeof value.data === "object") {
    return findMonitoringTaskId(value.data);
  }
  return "";
}

function findRealtimeTaskId(value) {
  if (!value || typeof value !== "object") return "";
  if (value.skill_name === "start-realtime-inspection" && value.data?.task_id) return String(value.data.task_id);
  if (Array.isArray(value.steps)) {
    for (let index = value.steps.length - 1; index >= 0; index -= 1) {
      const nested = findRealtimeTaskId(value.steps[index]);
      if (nested) return nested;
    }
  }
  if (value.data && typeof value.data === "object") return findRealtimeTaskId(value.data);
  return "";
}

function findRealtimeTask(value) {
  if (!value || typeof value !== "object") return null;
  if (value.task_id && value.status && value.source_id) return value;
  if (value.task && typeof value.task === "object") {
    const nestedTask = findRealtimeTask(value.task);
    if (nestedTask) return nestedTask;
  }
  if (Array.isArray(value.steps)) {
    for (let index = value.steps.length - 1; index >= 0; index -= 1) {
      const nested = findRealtimeTask(value.steps[index]);
      if (nested) return nested;
    }
  }
  if (Array.isArray(value.tasks) && value.tasks.length) {
    const active = value.tasks.find((item) => item && !TERMINAL_REALTIME_STATUSES.has(String(item.status || "")));
    return findRealtimeTask(active || value.tasks[0]);
  }
  if (value.data && typeof value.data === "object") return findRealtimeTask(value.data);
  return null;
}

function monitoringPathLabel(path) {
  const normalized = String(path || "").replaceAll("\\", "/");
  return normalized.split("/").pop() || "";
}

export function mountAgentChat(root) {
  const form = root.querySelector("[data-agent-form]");
  const textarea = form.querySelector("textarea[name='message']");
  const mediaInput = form.querySelector("input[name='media']");
  const sendButton = form.querySelector("button[type='submit']");
  const messages = root.querySelector("[data-agent-messages]");
  const status = root.querySelector("[data-agent-status]");
  const fileLabel = root.querySelector("[data-agent-file]");
  const monitoringPanel = root.querySelector("[data-agent-monitoring]");
  const monitoringConnection = root.querySelector("[data-agent-monitoring-connection]");
  const monitoringSegment = root.querySelector("[data-agent-monitoring-segment]");
  const monitoringProgress = root.querySelector("[data-agent-monitoring-progress]");
  const monitoringAlarm = root.querySelector("[data-agent-monitoring-alarm]");
  const monitoringReason = root.querySelector("[data-agent-monitoring-reason]");
  const monitoringProgressBar = root.querySelector("[data-agent-monitoring-progress-bar]");
  const monitoringStop = root.querySelector("[data-agent-monitoring-stop]");
  const endpoint = root.dataset.endpoint || "/api/agent/chat";
  const historyEndpoint = root.dataset.historyEndpoint || "/api/agent/history";
  const monitoringStopEndpoint = root.dataset.monitoringStopEndpoint
    || "/api/agent/monitoring/stop";
  const monitoringEventsEndpoint = root.dataset.monitoringEventsEndpoint
    || "/api/agent/monitoring/events";
  const realtimeStatusEndpoint = root.dataset.realtimeStatusEndpoint || "/api/agent/realtime-inspection/status";
  const realtimeEventsEndpoint = root.dataset.realtimeEventsEndpoint || "/api/agent/realtime-inspection/events";
  const storageKey = "foreign-object-agent-session";
  let sessionId = localStorage.getItem(storageKey);
  if (!sessionId) {
    sessionId = newSessionId();
    localStorage.setItem(storageKey, sessionId);
  }
  const monitoringTaskStorageKey = `${storageKey}:monitoring-task:${sessionId}`;
  let monitoringTaskId = localStorage.getItem(monitoringTaskStorageKey) || "";
  let monitoringCursor = "";
  let monitoringTimer = 0;
  let monitoringPolling = false;
  let lastMonitoringAlarmId = "";
  const realtimeTaskStorageKey = `${storageKey}:realtime-task:${sessionId}`;
  const realtimeReportStorageKey = `${storageKey}:realtime-report:${sessionId}`;
  let realtimeTaskId = localStorage.getItem(realtimeTaskStorageKey) || "";
  let realtimeReportAnnouncedTaskId = localStorage.getItem(realtimeReportStorageKey) || "";
  let realtimeTimer = 0;
  let realtimePolling = false;
  let realtimeEventTimer = 0;
  let realtimeEventPolling = false;
  let realtimeEventCursor = "";
  let displayedRealtimeEvents = new Set();
  let latestDetectionId = "";

  function setLatestDetectionId(detectionId) {
    latestDetectionId = String(detectionId || "").trim();
  }

  function append(role, text, isError = false, attachment = null) {
    const article = createMessage(role, text, isError, attachment);
    messages.appendChild(article);
    messages.scrollTop = messages.scrollHeight;
    return article;
  }

  function replaceAttachmentPreview(article, storedAttachment) {
    const preview = storedAttachmentPreview(storedAttachment);
    if (!article || !preview) return;
    const media = article.querySelector("img, video");
    if (!media) return;
    media.src = preview.src;
    if (media instanceof HTMLVideoElement) {
      media.poster = preview.poster;
      media.load();
    }
  }

  function appendAlarmReport(article, report) {
    if (!article || !report?.text) return;
    article.classList.add("agent-chat__message--with-report");
    const reportBody = document.createElement("div");
    reportBody.className = "agent-chat__report-body";
    appendReportContent(reportBody, report);
    if (report.jsonPath) {
      const link = document.createElement("a");
      link.className = "agent-chat__report-link";
      link.href = outputPathToUrl(report.jsonPath);
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = "查看报警 JSON";
      reportBody.append(link);
    }
    article.append(reportBody);
    messages.scrollTop = messages.scrollHeight;
  }

  function renderMonitoring(snapshot, { announceAlarm = true } = {}) {
    if (!snapshot?.found && !snapshot?.task_id) {
      monitoringPanel.hidden = true;
      return;
    }
    monitoringPanel.hidden = false;
    const connection = snapshot.connection || {};
    const segment = snapshot.current_segment || {};
    const progress = snapshot.progress || {};
    const alarm = snapshot.latest_alarm || {};
    monitoringConnection.textContent = connection.label || snapshot.status || "未知";
    monitoringSegment.textContent = segment.segment_id
      ? `${monitoringPathLabel(segment.video_path) || segment.segment_id} · ${segment.status || ""}`
      : "等待片段";
    const percent = Math.max(0, Math.min(100, Number(progress.estimated_percent || 0)));
    const phaseLabels = {
      waiting: "等待开始",
      capturing_or_detecting: "正在采集/检测",
      waiting_next_segment: "等待下一轮",
      stopping_after_current_segment: "当前轮结束后停止",
      completed: "检测完成",
      failed: "检测失败",
      cancelled: "已取消",
    };
    monitoringProgress.textContent = `${phaseLabels[progress.phase] || progress.phase || "—"} · ${percent}%`;
    monitoringProgressBar.style.width = `${percent}%`;
    monitoringAlarm.textContent = alarm.alarm_id
      ? `${alarm.risk_level || "未知风险"} · ${alarm.status || alarm.alarm_id}`
      : "暂无报警";
    monitoringReason.textContent = snapshot.stop_reason || "—";
    const terminal = TERMINAL_MONITORING_STATUSES.has(String(snapshot.status || ""));
    monitoringStop.hidden = terminal;
    monitoringStop.disabled = false;

    const alarmId = String(alarm.alarm_id || "");
    if (alarmId && alarmId !== lastMonitoringAlarmId) {
      if (announceAlarm) {
        const article = append(
          "assistant",
          `监控任务发现最新报警：${alarm.risk_level || "未知风险"}。`,
        );
        appendAlarmReport(article, extractAlarmReport(alarm));
      }
      lastMonitoringAlarmId = alarmId;
    }
  }

  function scheduleMonitoringPoll(delay = 2000) {
    window.clearTimeout(monitoringTimer);
    monitoringTimer = window.setTimeout(() => pollMonitoring(), delay);
  }

  async function pollMonitoring(initial = false) {
    if (monitoringPolling) return;
    monitoringPolling = true;
    try {
      const url = new URL(monitoringEventsEndpoint, window.location.origin);
      url.searchParams.set("session_id", sessionId);
      url.searchParams.set("limit", "50");
      if (monitoringTaskId) url.searchParams.set("task_id", monitoringTaskId);
      if (monitoringCursor) url.searchParams.set("after_segment_id", monitoringCursor);
      const response = await fetch(url);
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data.ok) {
        if (response.status === 404) {
          localStorage.removeItem(monitoringTaskStorageKey);
          monitoringTaskId = "";
          monitoringPanel.hidden = true;
        }
        return;
      }
      if (!data.found && !data.task_id) {
        monitoringPanel.hidden = true;
        return;
      }
      if (data.task_id) {
        monitoringTaskId = String(data.task_id);
        localStorage.setItem(monitoringTaskStorageKey, monitoringTaskId);
      }
      if (data.next_cursor) monitoringCursor = String(data.next_cursor);
      renderMonitoring(data, { announceAlarm: !initial });
      if (!TERMINAL_MONITORING_STATUSES.has(String(data.status || ""))) {
        scheduleMonitoringPoll();
      } else {
        localStorage.removeItem(monitoringTaskStorageKey);
      }
    } catch (_error) {
      scheduleMonitoringPoll(5000);
    } finally {
      monitoringPolling = false;
    }
  }

  function activateMonitoring(taskId) {
    if (!taskId) return;
    monitoringTaskId = taskId;
    monitoringCursor = "";
    lastMonitoringAlarmId = "";
    localStorage.setItem(monitoringTaskStorageKey, taskId);
    scheduleMonitoringPoll(0);
  }

  function renderRealtime(task, events = []) {
    if (!task?.task_id) return;
    events.forEach((event) => {
      const key = realtimeEventKey(event);
      messages.querySelectorAll("[data-realtime-event-key]").forEach((article) => {
        if (article.dataset.realtimeEventKey !== key) return;
        const statusNode = article.querySelector("[data-realtime-event-status]");
        if (statusNode) statusNode.textContent = event.event_status === "closed" ? "已关闭" : "持续中";
      });
    });
    document.dispatchEvent(new CustomEvent("agent:realtime-status", {
      detail: { task, events },
    }));
  }

  function realtimeEventStorage(taskId) {
    return {
      cursor: `${storageKey}:realtime-event-cursor:${sessionId}:${taskId}`,
      seen: `${storageKey}:realtime-events-seen:${sessionId}:${taskId}`,
    };
  }

  function loadRealtimeEventState(taskId) {
    const keys = realtimeEventStorage(taskId);
    realtimeEventCursor = localStorage.getItem(keys.cursor) || "";
    try {
      const values = JSON.parse(localStorage.getItem(keys.seen) || "[]");
      displayedRealtimeEvents = new Set(Array.isArray(values) ? values.map(String) : []);
    } catch (_error) {
      displayedRealtimeEvents = new Set();
    }
  }

  function saveRealtimeEventState(taskId) {
    const keys = realtimeEventStorage(taskId);
    if (realtimeEventCursor) localStorage.setItem(keys.cursor, realtimeEventCursor);
    localStorage.setItem(keys.seen, JSON.stringify(Array.from(displayedRealtimeEvents).slice(-200)));
  }

  function scheduleRealtimeEventPoll(delay = 3000) {
    window.clearTimeout(realtimeEventTimer);
    realtimeEventTimer = window.setTimeout(() => pollRealtimeEvents(), delay);
  }

  async function pollRealtimeEvents(initial = false, explicitTaskId = "") {
    if (realtimeEventPolling) return;
    const taskId = String(explicitTaskId || realtimeTaskId || "");
    if (!taskId) return;
    realtimeEventPolling = true;
    try {
      const url = new URL(realtimeEventsEndpoint, window.location.origin);
      url.searchParams.set("session_id", sessionId);
      url.searchParams.set("task_id", taskId);
      url.searchParams.set("limit", "50");
      if (realtimeEventCursor) url.searchParams.set("after_event_id", realtimeEventCursor);
      const response = await fetch(url);
      const result = await response.json().catch(() => ({}));
      if (!response.ok || !result.ok) return;
      const events = Array.isArray(result.data?.events) ? result.data.events : [];
      const task = result.data?.task || {};
      events.forEach((event) => {
        const key = realtimeEventKey(event);
        if (!displayedRealtimeEvents.has(key)) {
          const article = append("assistant", "检测到新的实时异物事件。", false);
          appendRealtimeEvent(article, event);
          displayedRealtimeEvents.add(key);
          if (event.detection_id) setLatestDetectionId(event.detection_id);
          document.dispatchEvent(new CustomEvent("agent:realtime-event", {
            detail: { event, task },
          }));
        }
        realtimeEventCursor = String(event.event_id || realtimeEventCursor);
      });
      if (result.data?.next_event_id) {
        realtimeEventCursor = String(result.data.next_event_id);
      }
      saveRealtimeEventState(taskId);
      if (task && !TERMINAL_REALTIME_STATUSES.has(String(task.status || ""))) {
        scheduleRealtimeEventPoll(initial ? 1000 : 3000);
      }
    } catch (_error) {
      if (realtimeTaskId) scheduleRealtimeEventPoll(5000);
    } finally {
      realtimeEventPolling = false;
    }
  }

  async function pollRealtime(initial = false) {
    if (realtimePolling) {
      window.clearTimeout(realtimeTimer);
      realtimeTimer = window.setTimeout(() => pollRealtime(initial), 250);
      return;
    }
    realtimePolling = true;
    try {
      const requestedTaskId = realtimeTaskId;
      const url = new URL(realtimeStatusEndpoint, window.location.origin);
      url.searchParams.set("session_id", sessionId);
      if (realtimeTaskId) url.searchParams.set("task_id", realtimeTaskId);
      const response = await fetch(url);
      const result = await response.json().catch(() => ({}));
      const task = result.data?.task || result.data?.tasks?.[0];
      if (response.ok && result.ok && task) {
        realtimeTaskId = String(task.task_id);
        localStorage.setItem(realtimeTaskStorageKey, realtimeTaskId);
        renderRealtime(task, result.data?.events || []);
        if (!TERMINAL_REALTIME_STATUSES.has(String(task.status || ""))) {
          realtimeTimer = window.setTimeout(() => pollRealtime(), 3000);
        } else {
          await pollRealtimeEvents(false, String(task.task_id));
          window.clearTimeout(realtimeEventTimer);
          localStorage.removeItem(realtimeTaskStorageKey);
          if (requestedTaskId && realtimeReportAnnouncedTaskId !== String(task.task_id)) {
            append("assistant", result.reply || "实时巡检已结束。", false);
            realtimeReportAnnouncedTaskId = String(task.task_id);
            localStorage.setItem(realtimeReportStorageKey, realtimeReportAnnouncedTaskId);
          }
        }
      }
    } catch (_error) {
      realtimeTimer = window.setTimeout(() => pollRealtime(), 5000);
    } finally { realtimePolling = false; }
  }

  function activateRealtime(taskId, task = null) {
    realtimeTaskId = taskId;
    loadRealtimeEventState(taskId);
    localStorage.setItem(realtimeTaskStorageKey, taskId);
    window.clearTimeout(realtimeTimer);
    if (task) renderRealtime(task);
    if (task && TERMINAL_REALTIME_STATUSES.has(String(task.status || ""))) {
      localStorage.removeItem(realtimeTaskStorageKey);
      return;
    }
    realtimeTimer = window.setTimeout(() => pollRealtime(), 0);
    scheduleRealtimeEventPoll(0);
  }

  async function stopMonitoring() {
    if (!monitoringTaskId || monitoringStop.disabled) return;
    monitoringStop.disabled = true;
    try {
      const response = await fetch(monitoringStopEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, task_id: monitoringTaskId }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data.ok) throw new Error(data.error || data.reply || "停止失败");
      append("assistant", data.reply || "已请求停止监控任务。");
      scheduleMonitoringPoll(0);
    } catch (error) {
      append("assistant", `停止监控失败：${error.message}`, true);
      monitoringStop.disabled = false;
    }
  }

  function setBusy(busy, busyText = "处理中") {
    textarea.disabled = busy;
    mediaInput.disabled = busy;
    sendButton.disabled = busy;
    status.textContent = busy ? busyText : "待命";
    status.classList.toggle("is-busy", busy);
    document.dispatchEvent(new CustomEvent("agent:status", {
      detail: { busy, text: busy ? busyText : "待命" },
    }));
  }

  async function loadHistory() {
    try {
      const url = new URL(historyEndpoint, window.location.origin);
      url.searchParams.set("session_id", sessionId);
      url.searchParams.set("limit", "50");
      const response = await fetch(url);
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data.ok || !Array.isArray(data.messages)) return;
      if (data.messages.length === 0) return;
      messages.replaceChildren();
      data.messages.forEach((item) => {
        const role = item.role === "user" ? "user" : "assistant";
        const storedAttachment = item.metadata?.attachment;
        const attachment = storedAttachmentPreview(storedAttachment);
        let content = String(item.content || "");
        if (attachment && /^已发送(?:图片|视频)(?:：.*)?$/.test(content)) {
          content = "";
        }
        const article = append(role, content, false, attachment);
        if (role === "assistant") {
          const realtimeReport = extractRealtimeReport(item.metadata?.data);
          const presentation = extractDetectionPresentation(item.metadata?.data);
          if (realtimeReport) {
            appendRealtimeReport(article, realtimeReport);
          } else if (presentation) {
            appendDetectionPresentation(article, presentation);
          } else {
            appendAlarmReport(article, extractAlarmReport(item.metadata?.data));
          }
          const detectionId = findDetectionId(item.metadata?.data);
          if (detectionId) setLatestDetectionId(detectionId);
        }
      });
    } catch (_error) {
      // History is helpful but non-blocking; chat remains usable if it fails.
    }
  }

  async function submit() {
    const message = textarea.value.trim();
    const media = mediaInput.files?.[0];
    if (!message && !media) return;
    const isImage = media && (
      media.type?.startsWith("image/")
      || /\.(?:jpe?g|png|bmp|webp)$/i.test(media.name)
    );
    const attachment = media ? {
      media_type: isImage ? "image" : "video",
      src: URL.createObjectURL(media),
    } : null;
    const userArticle = append("user", message, false, attachment);
    const realtimeStartRequested = Boolean(message) && /(?:开始|启动|开启|安排|从现在开始|从.+开始).{0,30}(?:实时巡检|持续巡检|持续实时|持续连接检测)/.test(message);
    const realtimeControlRequested = Boolean(message) && /(?:查看|查询|显示|停止|终止|结束|取消).{0,24}(?:实时巡检|持续巡检)|(?:实时巡检|持续巡检).{0,24}(?:状态|停止|终止|结束|取消)/.test(message);
    const monitoringStartRequested = !realtimeStartRequested && Boolean(message) && (
      /(?:开始|启动|开启|创建|安排|预约).{0,16}(?:监控|巡检)/.test(message)
      || /(?:立即|现在)(?:开始)?(?:监控|巡检)/.test(message)
      || /(?:监控|巡检).*(?:从|到|至|持续|分钟|小时|今天|明天)/.test(message)
    );
    const monitoringControlRequested = Boolean(message) && (
      /(?:查看|查询|显示|获取|停止|终止|结束|取消|关闭).{0,16}(?:监控|巡检|任务)/.test(message)
      || /(?:监控|巡检|任务).{0,16}(?:状态|停止|终止|结束|取消|关闭)/.test(message)
    );
    const detectionRequested = Boolean(message) && !monitoringStartRequested && (
      Boolean(media)
      || (
        /(?:检测|识别|分析|检查|巡检|看看|看一下|跑一下)/.test(message)
        && /(?:图片|图像|照片|视频|这张|这段|这个|监控|摄像头|视频流|实时流|RTSP|monitor)/i.test(message)
      )
    );
    const pendingText = realtimeStartRequested
      ? "正在启动实时巡检..."
      : realtimeControlRequested
        ? "正在处理实时巡检任务..."
      : monitoringStartRequested
      ? "正在创建监控任务..."
      : monitoringControlRequested
        ? "正在处理监控任务..."
        : detectionRequested
      ? "正在检测..."
      : media && !message
        ? `正在接收${isImage ? "图片" : "视频"}...`
        : "正在处理...";
    const pendingArticle = append("assistant", pendingText);
    pendingArticle.classList.add("agent-chat__message--pending");
    const body = new FormData();
    body.append("message", message);
    body.append("session_id", sessionId);
    if (latestDetectionId) body.append("detection_id", latestDetectionId);
    if (realtimeTaskId) body.append("task_id", realtimeTaskId);
    if (media) body.append("media", media);
    textarea.value = "";
    textarea.style.height = "auto";
    setBusy(true, pendingText.replace(/\.{3}$/, ""));

    try {
      const response = await fetch(endpoint, { method: "POST", body });
      const data = await response.json().catch(() => ({}));
      pendingArticle.remove();
      if (!response.ok) {
        throw new Error(data.error || `智能体接口返回 ${response.status}`);
      }
      if (media && data.attachment) {
        replaceAttachmentPreview(userArticle, data.attachment);
      }
      const assistantArticle = append(
        "assistant",
        data.reply || "操作已完成。",
        !data.ok,
      );
      const realtimeReport = extractRealtimeReport(data.data);
      const presentation = extractDetectionPresentation(data.data);
      if (realtimeReport) {
        appendRealtimeReport(assistantArticle, realtimeReport);
        realtimeReportAnnouncedTaskId = String(realtimeReport.task_id);
        localStorage.setItem(realtimeReportStorageKey, realtimeReportAnnouncedTaskId);
      } else if (presentation) {
        appendDetectionPresentation(assistantArticle, presentation);
      } else {
        appendAlarmReport(assistantArticle, extractAlarmReport(data.data));
      }
      const detectionId = findDetectionId(data.data);
      if (detectionId) setLatestDetectionId(detectionId);
      const monitoringId = findMonitoringTaskId(data);
      if (monitoringId) activateMonitoring(monitoringId);
      const realtimeTask = findRealtimeTask(data);
      const realtimeId = String(realtimeTask?.task_id || findRealtimeTaskId(data) || "");
      if (realtimeId) activateRealtime(realtimeId, realtimeTask);
      document.dispatchEvent(new CustomEvent("agent:response", { detail: { data } }));
      if (data.attachment_received) {
        mediaInput.value = "";
        fileLabel.textContent = "";
      }
    } catch (error) {
      pendingArticle.remove();
      const pending = error.message.includes("404")
        ? "聊天组件已就绪，后端 /api/agent/chat 尚待接入 web_app.py。"
        : error.message;
      append("assistant", pending, true);
    } finally {
      setBusy(false);
      textarea.focus();
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    submit();
  });

  textarea.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  textarea.addEventListener("input", () => {
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 132)}px`;
  });

  mediaInput.addEventListener("change", () => {
    const file = mediaInput.files?.[0];
    fileLabel.textContent = file ? `已选择：${file.name}` : "";
  });

  monitoringStop?.addEventListener("click", stopMonitoring);

  root.addEventListener("click", (event) => {
    const button = event.target.closest("[data-agent-alarm-action]");
    if (!button || !root.contains(button)) return;
    const action = button.dataset.agentAlarmAction;
    const taskId = String(button.dataset.taskId || "");
    if (!taskId || !["confirm", "cancel"].includes(action)) return;
    realtimeTaskId = taskId;
    const prompt = action === "confirm" ? "确认本轮报警" : "取消本轮报警";
    button.closest(".agent-chat__alarm-actions")?.querySelectorAll("button")
      .forEach((item) => { item.disabled = true; });
    textarea.value = prompt;
    textarea.dispatchEvent(new Event("input"));
    form.requestSubmit();
  });

  document.addEventListener("agent:prefill", (event) => {
    const prompt = String(event.detail?.prompt || "").trim();
    if (!prompt) return;
    textarea.value = prompt;
    textarea.dispatchEvent(new Event("input"));
    textarea.focus();
  });

  if (realtimeTaskId) {
    loadRealtimeEventState(realtimeTaskId);
    scheduleRealtimeEventPoll(0);
  }
  loadHistory();
  pollMonitoring(true);
  pollRealtime(true);
}

document.querySelectorAll("[data-agent-chat]").forEach(mountAgentChat);
