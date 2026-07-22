function createMessage(role, text, isError = false, attachment = null) {
  const article = document.createElement("article");
  article.className = `agent-chat__message agent-chat__message--${role}`;
  article.setAttribute("aria-label", role === "user" ? "你的消息" : "智能体消息");
  if (isError) article.classList.add("agent-chat__message--error");
  if (attachment?.src) article.classList.add("agent-chat__message--media");

  if (text) {
    const content = document.createElement("p");
    content.textContent = text;
    article.append(content);
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

function displayAlarmReportText(text) {
  let report = String(text || "").trim();
  const conclusionIndex = report.indexOf("二、报警结论");
  if (conclusionIndex >= 0) report = report.slice(conclusionIndex);
  const generationIndex = report.indexOf("\n七、生成信息");
  if (generationIndex >= 0) report = report.slice(0, generationIndex);
  const sectionNumbers = [
    ["二、报警结论", "一、报警结论"],
    ["三、总体风险等级", "二、总体风险等级"],
    ["四、事件详情", "三、事件详情"],
    ["五、风险说明", "四、风险说明"],
    ["六、处理建议", "五、处理建议"],
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
  const riskStart = text.indexOf("\n四、风险说明", detailStart);
  if (frameByEvent.size === 0 || detailStart < 0 || riskStart < 0) {
    appendReportText(container, text);
    return;
  }

  appendReportText(container, text.slice(0, detailStart));
  const detailText = text.slice(detailStart, riskStart);
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
  appendReportText(container, text.slice(riskStart));
}

function newSessionId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

const TERMINAL_MONITORING_STATUSES = new Set(["completed", "failed", "cancelled"]);

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
          appendAlarmReport(article, extractAlarmReport(item.metadata?.data));
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
    const monitoringStartRequested = Boolean(message) && (
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
    const pendingText = monitoringStartRequested
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
      appendAlarmReport(assistantArticle, extractAlarmReport(data.data));
      const monitoringId = findMonitoringTaskId(data);
      if (monitoringId) activateMonitoring(monitoringId);
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

  root.querySelectorAll("[data-agent-prompt]").forEach((button) => {
    button.addEventListener("click", () => {
      textarea.value = button.dataset.agentPrompt || "";
      textarea.focus();
    });
  });

  document.addEventListener("agent:prefill", (event) => {
    const prompt = String(event.detail?.prompt || "").trim();
    if (!prompt) return;
    textarea.value = prompt;
    textarea.dispatchEvent(new Event("input"));
    textarea.focus();
  });

  loadHistory();
  pollMonitoring(true);
}

document.querySelectorAll("[data-agent-chat]").forEach(mountAgentChat);
