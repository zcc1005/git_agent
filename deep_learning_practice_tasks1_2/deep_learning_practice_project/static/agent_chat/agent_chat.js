function createMessage(role, text, isError = false, attachment = null) {
  const article = document.createElement("article");
  article.className = `agent-chat__message agent-chat__message--${role}`;
  if (isError) article.classList.add("agent-chat__message--error");
  if (attachment?.src) article.classList.add("agent-chat__message--media");

  const label = document.createElement("span");
  label.textContent = role === "user" ? "你" : "助手";
  article.append(label);
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

export function mountAgentChat(root) {
  const form = root.querySelector("[data-agent-form]");
  const textarea = form.querySelector("textarea[name='message']");
  const mediaInput = form.querySelector("input[name='media']");
  const sendButton = form.querySelector("button[type='submit']");
  const messages = root.querySelector("[data-agent-messages]");
  const status = root.querySelector("[data-agent-status]");
  const fileLabel = root.querySelector("[data-agent-file]");
  const endpoint = root.dataset.endpoint || "/api/agent/chat";
  const historyEndpoint = root.dataset.historyEndpoint || "/api/agent/history";
  const storageKey = "foreign-object-agent-session";
  let sessionId = localStorage.getItem(storageKey);
  if (!sessionId) {
    sessionId = newSessionId();
    localStorage.setItem(storageKey, sessionId);
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

  function setBusy(busy, busyText = "处理中") {
    textarea.disabled = busy;
    mediaInput.disabled = busy;
    sendButton.disabled = busy;
    status.textContent = busy ? busyText : "待命";
    status.classList.toggle("is-busy", busy);
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
    const detectionRequested = Boolean(message) && (
      Boolean(media)
      || (
        /(?:检测|识别|分析|检查|看看|看一下|跑一下)/.test(message)
        && /(?:图片|图像|照片|视频|这张|这段|这个)/.test(message)
      )
    );
    const pendingText = detectionRequested
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

  root.querySelectorAll("[data-agent-prompt]").forEach((button) => {
    button.addEventListener("click", () => {
      textarea.value = button.dataset.agentPrompt || "";
      textarea.focus();
    });
  });

  loadHistory();
}

document.querySelectorAll("[data-agent-chat]").forEach(mountAgentChat);
