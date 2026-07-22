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
  if (root.dataset.agentChatMounted === "true") return;
  root.dataset.agentChatMounted = "true";
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
  const realtimeCard = root.querySelector("[data-agent-realtime-card]");
  const realtimeToggle = root.querySelector("[data-agent-realtime-toggle]");
  const realtimeDetail = root.querySelector("[data-agent-realtime-detail]");
  const realtimeTitle = root.querySelector("[data-agent-realtime-title]");
  const realtimeSummary = root.querySelector("[data-agent-realtime-summary]");
  const realtimeDot = root.querySelector("[data-agent-realtime-dot]");
  const realtimeUnread = root.querySelector("[data-agent-realtime-unread]");
  const realtimeStatus = root.querySelector("[data-agent-realtime-status]");
  const realtimeDuration = root.querySelector("[data-agent-realtime-duration]");
  const realtimeFramesRead = root.querySelector("[data-agent-realtime-frames-read]");
  const realtimeFramesInferred = root.querySelector("[data-agent-realtime-frames-inferred]");
  const realtimeEventCount = root.querySelector("[data-agent-realtime-event-count]");
  const realtimePendingCount = root.querySelector("[data-agent-realtime-pending-count]");
  const realtimeRisk = root.querySelector("[data-agent-realtime-risk]");
  const realtimeLatest = root.querySelector("[data-agent-realtime-latest]");
  const realtimeEvents = root.querySelector("[data-agent-realtime-events]");
  const realtimeTerminal = root.querySelector("[data-agent-realtime-terminal]");
  const realtimeStop = root.querySelector("[data-agent-realtime-stop]");
  const realtimeConfirm = root.querySelector("[data-agent-realtime-confirm]");
  const realtimeCancel = root.querySelector("[data-agent-realtime-cancel]");
  const sessionTabs = root.querySelector("[data-agent-session-tabs]");
  const newSessionButton = root.querySelector("[data-agent-session-new]");
  const endpoint = root.dataset.endpoint || "/api/agent/chat";
  const historyEndpoint = root.dataset.historyEndpoint || "/api/agent/history";
  const monitoringStopEndpoint = root.dataset.monitoringStopEndpoint
    || "/api/agent/monitoring/stop";
  const monitoringEventsEndpoint = root.dataset.monitoringEventsEndpoint
    || "/api/agent/monitoring/events";
  const realtimeStatusEndpoint = root.dataset.realtimeStatusEndpoint || "/api/agent/realtime-inspection/status";
  const realtimeEventsEndpoint = root.dataset.realtimeEventsEndpoint || "/api/agent/realtime-inspection/events";
  const realtimeStopEndpoint = root.dataset.realtimeStopEndpoint || "/api/agent/realtime-inspection/stop";
  const storageKey = "foreign-object-agent-session";
  const sessionsStorageKey = `${storageKey}:sessions-v1`;
  const activeRealtimeTaskStorageKey = `${storageKey}:active-realtime-task`;
  let sessions = [];
  try {
    const stored = JSON.parse(localStorage.getItem(sessionsStorageKey) || "[]");
    if (Array.isArray(stored)) sessions = stored.filter((item) => item?.id);
  } catch (_error) {
    sessions = [];
  }
  let sessionId = localStorage.getItem(storageKey) || "";
  if (!sessions.length) {
    sessionId = sessionId || newSessionId();
    sessions = [{ id: sessionId, title: "对话 1" }];
  } else if (!sessions.some((item) => item.id === sessionId)) {
    sessionId = sessions[0].id;
  }
  localStorage.setItem(storageKey, sessionId);
  localStorage.setItem(sessionsStorageKey, JSON.stringify(sessions));
  const monitoringTaskStorageKey = () => `${storageKey}:monitoring-task:${sessionId}`;
  let monitoringTaskId = localStorage.getItem(monitoringTaskStorageKey()) || "";
  let monitoringCursor = "";
  let monitoringTimer = 0;
  let monitoringPolling = false;
  let lastMonitoringAlarmId = "";
  let activeTask = {};
  try {
    activeTask = JSON.parse(localStorage.getItem(activeRealtimeTaskStorageKey) || "{}") || {};
  } catch (_error) {
    activeTask = {};
  }
  let realtimeTaskId = String(activeTask.active_task_id || activeTask.task_id || "");
  let realtimeTaskOwnerSessionId = String(activeTask.session_id || "");
  let realtimeTimer = 0;
  let realtimePolling = false;
  let realtimeEventTimer = 0;
  let realtimeEventPolling = false;
  let realtimeEventCursor = "";
  let displayedRealtimeEvents = new Set();
  let highRiskNotifiedEvents = new Set();
  let unreadRealtimeEvents = 0;
  const realtimeEventsByKey = new Map();
  const terminalAnnouncedTasks = new Set();
  let latestDetectionId = "";
  let historyLoadToken = 0;

  function setLatestDetectionId(detectionId) {
    latestDetectionId = String(detectionId || "").trim();
  }

  function append(role, text, isError = false, attachment = null) {
    const article = createMessage(role, text, isError, attachment);
    messages.appendChild(article);
    messages.scrollTop = messages.scrollHeight;
    return article;
  }

  function persistSessions() {
    localStorage.setItem(sessionsStorageKey, JSON.stringify(sessions.slice(0, 12)));
    localStorage.setItem(storageKey, sessionId);
  }

  function renderSessionTabs() {
    sessionTabs.replaceChildren();
    sessions.forEach((item) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "agent-chat__session-tab";
      button.classList.toggle("is-active", item.id === sessionId);
      button.dataset.sessionId = item.id;
      button.append(document.createTextNode(item.title || "对话"));
      const close = document.createElement("span");
      close.dataset.sessionClose = item.id;
      close.title = "关闭对话";
      close.textContent = "×";
      button.append(close);
      sessionTabs.append(button);
    });
  }

  function setSessionTitle(message) {
    const current = sessions.find((item) => item.id === sessionId);
    if (!current || !/^对话\s*\d+$/.test(current.title || "")) return;
    const clean = String(message || "").replace(/\s+/g, " ").trim();
    if (!clean) return;
    current.title = clean.length > 12 ? `${clean.slice(0, 12)}…` : clean;
    persistSessions();
    renderSessionTabs();
  }

  function switchSession(nextSessionId) {
    if (!nextSessionId || nextSessionId === sessionId || sendButton.disabled) return;
    sessionId = nextSessionId;
    persistSessions();
    renderSessionTabs();
    latestDetectionId = "";
    mediaInput.value = "";
    fileLabel.textContent = "";
    messages.replaceChildren();
    window.clearTimeout(monitoringTimer);
    monitoringTaskId = localStorage.getItem(monitoringTaskStorageKey()) || "";
    monitoringCursor = "";
    loadHistory();
    pollMonitoring(true);
  }

  function createSession() {
    if (sendButton.disabled) return;
    const id = newSessionId();
    sessions.push({ id, title: `对话 ${sessions.length + 1}` });
    sessionId = id;
    persistSessions();
    renderSessionTabs();
    latestDetectionId = "";
    messages.replaceChildren();
    append("assistant", "新对话已创建。实时巡检任务仍在后台运行。", false);
    textarea.focus();
  }

  function closeSession(targetSessionId) {
    if (sendButton.disabled) return;
    sessions = sessions.filter((item) => item.id !== targetSessionId);
    if (!sessions.length) {
      const id = newSessionId();
      sessions = [{ id, title: "对话 1" }];
    }
    if (sessionId === targetSessionId) {
      sessionId = sessions[0].id;
      messages.replaceChildren();
      latestDetectionId = "";
      loadHistory();
    }
    persistSessions();
    renderSessionTabs();
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
          localStorage.removeItem(monitoringTaskStorageKey());
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
        localStorage.setItem(monitoringTaskStorageKey(), monitoringTaskId);
      }
      if (data.next_cursor) monitoringCursor = String(data.next_cursor);
      renderMonitoring(data, { announceAlarm: !initial });
      if (!TERMINAL_MONITORING_STATUSES.has(String(data.status || ""))) {
        scheduleMonitoringPoll();
      } else {
        localStorage.removeItem(monitoringTaskStorageKey());
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
    localStorage.setItem(monitoringTaskStorageKey(), taskId);
    scheduleMonitoringPoll(0);
  }

  function realtimeStatusName(value) {
    return {
      scheduled: "等待开始",
      connecting: "正在连接",
      running: "运行中",
      reconnecting: "正在重连",
      stop_requested: "正在停止",
      completed: "已结束",
      stopped: "已停止",
      failed: "异常结束",
      interrupted: "意外中断",
    }[String(value || "")] || String(value || "未知");
  }

  function realtimeRiskName(value) {
    return { high: "高风险", medium: "中风险", low: "低风险", none: "无风险" }[
      String(value || "none").toLowerCase()
    ] || String(value || "无风险");
  }

  function realtimeElapsed(task) {
    const start = new Date(task?.started_at || task?.start_time || "");
    const terminal = TERMINAL_REALTIME_STATUSES.has(String(task?.status || ""));
    const end = new Date(terminal ? (task?.stopped_at || task?.end_time || "") : Date.now());
    if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) return "—";
    const seconds = Math.max(0, Math.floor((end.getTime() - start.getTime()) / 1000));
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const rest = seconds % 60;
    return hours ? `${hours}小时${minutes}分` : minutes ? `${minutes}分${rest}秒` : `${rest}秒`;
  }

  function realtimeReportForEvent(event) {
    if (typeof event?.alarm_report === "string") {
      return { text: event.alarm_report, eventFrames: [] };
    }
    if (event?.alarm_report?.text) {
      return { text: String(event.alarm_report.text), eventFrames: [] };
    }
    return null;
  }

  function renderRealtimeEventCenter() {
    realtimeEvents.replaceChildren();
    const events = [...realtimeEventsByKey.values()].sort((left, right) => (
      String(right.detected_at || right.created_at || "").localeCompare(
        String(left.detected_at || left.created_at || ""),
      )
    ));
    if (!events.length) {
      const empty = document.createElement("p");
      empty.className = "agent-chat__realtime-empty";
      empty.textContent = "当前没有确认异物事件。";
      realtimeEvents.append(empty);
      return;
    }
    events.forEach((event) => {
      const article = createMessage("assistant", "", false);
      article.dataset.realtimeEventKey = realtimeEventKey(event);
      appendRealtimeEvent(article, event);
      const report = realtimeReportForEvent(event);
      if (report?.text) {
        const body = document.createElement("div");
        body.className = "agent-chat__report-body";
        appendReportContent(body, report);
        article.append(body);
      }
      realtimeEvents.append(article);
    });
  }

  function updateRealtimeUnread() {
    realtimeUnread.hidden = unreadRealtimeEvents <= 0;
    realtimeUnread.textContent = String(unreadRealtimeEvents);
  }

  function ingestRealtimeEvents(
    events,
    { markUnread = false, notifyHighRisk = false, forceNew = false } = {},
  ) {
    let changed = false;
    events.forEach((event) => {
      if (!event?.event_id || !event?.task_id) return;
      const key = realtimeEventKey(event);
      const isNew = forceNew || !realtimeEventsByKey.has(key);
      realtimeEventsByKey.set(key, event);
      changed = true;
      if (isNew && markUnread && realtimeCard.classList.contains("is-collapsed")) {
        unreadRealtimeEvents += 1;
      }
      if (
        isNew
        && notifyHighRisk
        && String(event.risk_level || "").toLowerCase() === "high"
        && !highRiskNotifiedEvents.has(key)
      ) {
        append("assistant", "实时巡检发现新的高风险报警，请展开顶部任务卡查看并处置。", false);
        highRiskNotifiedEvents.add(key);
      }
      if (event.detection_id) setLatestDetectionId(event.detection_id);
      document.dispatchEvent(new CustomEvent("agent:realtime-event", {
        detail: { event, task: activeTask.snapshot || {} },
      }));
    });
    if (changed) renderRealtimeEventCenter();
    updateRealtimeUnread();
  }

  function renderRealtime(task, events = []) {
    if (!task?.task_id) return;
    activeTask.snapshot = task;
    ingestRealtimeEvents(events);
    const terminal = TERMINAL_REALTIME_STATUSES.has(String(task.status || ""));
    const source = String(task.display_name || task.source_id || "监控源");
    const pending = [...realtimeEventsByKey.values()].filter(
      (event) => String(event.alarm_status || "") === "pending",
    ).length;
    const latestEvent = [...realtimeEventsByKey.values()].sort((left, right) => (
      String(right.last_seen_at || right.detected_at || "").localeCompare(
        String(left.last_seen_at || left.detected_at || ""),
      )
    ))[0];
    const risk = String(task.highest_risk_level || latestEvent?.risk_level || "none");
    realtimeTitle.textContent = terminal ? "实时巡检已结束" : `正在巡检：${source}`;
    realtimeSummary.textContent = `${realtimeStatusName(task.status)} · ${realtimeElapsed(task)} · ${Number(task.events_detected || realtimeEventsByKey.size)}个事件 · ${realtimeRiskName(risk)}`;
    realtimeStatus.textContent = realtimeStatusName(task.status);
    realtimeDuration.textContent = realtimeElapsed(task);
    realtimeFramesRead.textContent = String(Number(task.frames_read || 0));
    realtimeFramesInferred.textContent = String(Number(task.frames_inferred || 0));
    realtimeEventCount.textContent = String(Number(task.events_detected || realtimeEventsByKey.size));
    realtimePendingCount.textContent = String(pending);
    realtimeRisk.textContent = realtimeRiskName(risk);
    realtimeLatest.textContent = latestEvent
      ? `${latestEvent.class_name || "未知异物"} · ${latestEvent.detected_at || ""}`
      : "暂无";
    realtimeDot.className = `agent-chat__realtime-dot ${terminal ? "" : risk === "high" ? "is-warning" : "is-running"}`.trim();
    realtimeStop.hidden = terminal;
    realtimeConfirm.hidden = pending <= 0;
    realtimeCancel.hidden = pending <= 0;
    if (terminal) {
      const reason = task.status === "completed"
        ? "达到计划结束时间"
        : task.status === "stopped"
          ? "用户主动停止"
          : task.last_error_message || realtimeStatusName(task.status);
      realtimeTerminal.hidden = false;
      realtimeTerminal.textContent = `实时巡检已结束：${source}，运行${realtimeElapsed(task)}，读取${Number(task.frames_read || 0)}帧，推理${Number(task.frames_inferred || 0)}帧，确认${Number(task.events_detected || realtimeEventsByKey.size)}个事件。结束原因：${reason}。`;
      terminalAnnouncedTasks.add(String(task.task_id));
    } else {
      realtimeTerminal.hidden = true;
    }
    document.dispatchEvent(new CustomEvent("agent:realtime-status", {
      detail: { task, events: [...realtimeEventsByKey.values()] },
    }));
  }

  function realtimeEventStorage(taskId) {
    return {
      cursor: `${storageKey}:realtime-event-cursor:${taskId}`,
      seen: `${storageKey}:realtime-events-seen:${taskId}`,
      notified: `${storageKey}:realtime-high-risk-notified:${taskId}`,
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
    try {
      const values = JSON.parse(localStorage.getItem(keys.notified) || "[]");
      highRiskNotifiedEvents = new Set(Array.isArray(values) ? values.map(String) : []);
    } catch (_error) {
      highRiskNotifiedEvents = new Set();
    }
    realtimeEventsByKey.clear();
    unreadRealtimeEvents = 0;
    updateRealtimeUnread();
    renderRealtimeEventCenter();
  }

  function saveRealtimeEventState(taskId) {
    const keys = realtimeEventStorage(taskId);
    if (realtimeEventCursor) localStorage.setItem(keys.cursor, realtimeEventCursor);
    localStorage.setItem(keys.seen, JSON.stringify(Array.from(displayedRealtimeEvents).slice(-200)));
    localStorage.setItem(keys.notified, JSON.stringify(Array.from(highRiskNotifiedEvents).slice(-200)));
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
      url.searchParams.set("session_id", realtimeTaskOwnerSessionId || sessionId);
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
          displayedRealtimeEvents.add(key);
          ingestRealtimeEvents([event], {
            markUnread: true,
            notifyHighRisk: true,
            forceNew: true,
          });
        } else {
          ingestRealtimeEvents([event]);
        }
        realtimeEventCursor = String(event.event_id || realtimeEventCursor);
      });
      if (result.data?.next_event_id) {
        realtimeEventCursor = String(result.data.next_event_id);
      }
      saveRealtimeEventState(taskId);
      if (task?.task_id) renderRealtime(task, events);
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
      const url = new URL(realtimeStatusEndpoint, window.location.origin);
      url.searchParams.set("session_id", realtimeTaskOwnerSessionId || sessionId);
      if (realtimeTaskId) url.searchParams.set("task_id", realtimeTaskId);
      const response = await fetch(url);
      const result = await response.json().catch(() => ({}));
      const task = result.data?.task || result.data?.tasks?.[0];
      if (response.ok && result.ok && task) {
        if (!realtimeTaskId) {
          activateRealtime(String(task.task_id), task, realtimeTaskOwnerSessionId || sessionId);
          return;
        }
        renderRealtime(task, result.data?.events || []);
        if (!TERMINAL_REALTIME_STATUSES.has(String(task.status || ""))) {
          realtimeTimer = window.setTimeout(() => pollRealtime(), 3000);
        } else {
          await pollRealtimeEvents(false, String(task.task_id));
          window.clearTimeout(realtimeEventTimer);
          terminalAnnouncedTasks.add(String(task.task_id));
        }
      }
    } catch (_error) {
      realtimeTimer = window.setTimeout(() => pollRealtime(), 5000);
    } finally { realtimePolling = false; }
  }

  function activateRealtime(taskId, task = null, ownerSessionId = sessionId) {
    const changedTask = taskId !== realtimeTaskId;
    realtimeTaskId = taskId;
    realtimeTaskOwnerSessionId = ownerSessionId || sessionId;
    activeTask = {
      active_task_id: taskId,
      task_id: taskId,
      session_id: realtimeTaskOwnerSessionId,
      snapshot: task || activeTask.snapshot || null,
    };
    localStorage.setItem(activeRealtimeTaskStorageKey, JSON.stringify({
      active_task_id: taskId,
      task_id: taskId,
      session_id: realtimeTaskOwnerSessionId,
    }));
    if (changedTask) loadRealtimeEventState(taskId);
    window.clearTimeout(realtimeTimer);
    if (task) renderRealtime(task);
    if (task && TERMINAL_REALTIME_STATUSES.has(String(task.status || ""))) {
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

  async function stopRealtimeInspection() {
    if (!realtimeTaskId || realtimeStop.disabled) return;
    realtimeStop.disabled = true;
    try {
      const response = await fetch(realtimeStopEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: realtimeTaskOwnerSessionId || sessionId,
          task_id: realtimeTaskId,
        }),
      });
      const result = await response.json().catch(() => ({}));
      if (!response.ok || !result.ok) {
        throw new Error(result.error || result.reply || "停止实时巡检失败");
      }
      const task = result.data?.task || result.data;
      if (task?.task_id) renderRealtime(task, []);
      window.clearTimeout(realtimeTimer);
      realtimeTimer = window.setTimeout(() => pollRealtime(), 300);
    } catch (error) {
      realtimeTerminal.hidden = false;
      realtimeTerminal.textContent = `停止任务失败：${error.message}`;
    } finally {
      realtimeStop.disabled = false;
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
    const requestedSessionId = sessionId;
    const token = ++historyLoadToken;
    try {
      const url = new URL(historyEndpoint, window.location.origin);
      url.searchParams.set("session_id", requestedSessionId);
      url.searchParams.set("limit", "50");
      const response = await fetch(url);
      const data = await response.json().catch(() => ({}));
      if (token !== historyLoadToken || requestedSessionId !== sessionId) return;
      if (!response.ok || !data.ok || !Array.isArray(data.messages)) return;
      messages.replaceChildren();
      if (data.messages.length === 0) {
        append("assistant", "我已待命。你可以上传图片或视频，也可以直接查询风险和历史记录。", false);
        return;
      }
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
          if (!realtimeReport && presentation) {
            appendDetectionPresentation(article, presentation);
          } else if (!realtimeReport) {
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
    if (message) setSessionTitle(message);
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
    if (realtimeTaskId && realtimeTaskOwnerSessionId) {
      body.append("task_session_id", realtimeTaskOwnerSessionId);
    }
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
      if (!realtimeReport && presentation) {
        appendDetectionPresentation(assistantArticle, presentation);
      } else if (!realtimeReport) {
        appendAlarmReport(assistantArticle, extractAlarmReport(data.data));
      }
      const detectionId = findDetectionId(data.data);
      if (detectionId) setLatestDetectionId(detectionId);
      const monitoringId = findMonitoringTaskId(data);
      if (monitoringId) activateMonitoring(monitoringId);
      const realtimeTask = findRealtimeTask(data);
      const realtimeId = String(realtimeTask?.task_id || findRealtimeTaskId(data) || "");
      if (realtimeId) activateRealtime(realtimeId, realtimeTask, sessionId);
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
  realtimeStop?.addEventListener("click", stopRealtimeInspection);
  realtimeToggle?.addEventListener("click", () => {
    const collapsed = !realtimeCard.classList.contains("is-collapsed");
    realtimeCard.classList.toggle("is-collapsed", collapsed);
    realtimeDetail.hidden = collapsed;
    realtimeToggle.setAttribute("aria-expanded", String(!collapsed));
    if (!collapsed) {
      unreadRealtimeEvents = 0;
      updateRealtimeUnread();
    }
  });
  realtimeConfirm?.addEventListener("click", () => {
    if (!realtimeTaskId) return;
    textarea.value = "确认本轮报警";
    textarea.dispatchEvent(new Event("input"));
    form.requestSubmit();
  });
  realtimeCancel?.addEventListener("click", () => {
    if (!realtimeTaskId) return;
    textarea.value = "取消本轮报警";
    textarea.dispatchEvent(new Event("input"));
    form.requestSubmit();
  });
  newSessionButton?.addEventListener("click", createSession);
  sessionTabs?.addEventListener("click", (event) => {
    const close = event.target.closest("[data-session-close]");
    if (close) {
      event.stopPropagation();
      closeSession(close.dataset.sessionClose);
      return;
    }
    const tab = event.target.closest("[data-session-id]");
    if (tab) switchSession(tab.dataset.sessionId);
  });

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
  renderSessionTabs();
  loadHistory();
  pollMonitoring(true);
  pollRealtime(true);
}

document.querySelectorAll("[data-agent-chat]").forEach(mountAgentChat);
