function createMessage(role, text, isError = false) {
  const article = document.createElement("article");
  article.className = `agent-chat__message agent-chat__message--${role}`;
  if (isError) article.classList.add("agent-chat__message--error");

  const label = document.createElement("span");
  label.textContent = role === "user" ? "你" : "助手";
  const content = document.createElement("p");
  content.textContent = text;
  article.append(label, content);
  return article;
}

function outputPathToUrl(path) {
  const normalized = String(path || "").replaceAll("\\", "/");
  const relative = normalized.startsWith("outputs/")
    ? normalized.slice("outputs/".length)
    : normalized;
  return `/outputs/${relative.split("/").map(encodeURIComponent).join("/")}`;
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

  function append(role, text, isError = false) {
    messages.appendChild(createMessage(role, text, isError));
    messages.scrollTop = messages.scrollHeight;
  }

  function appendResultImage(path) {
    if (!path) return;
    const article = document.createElement("article");
    article.className = "agent-chat__message agent-chat__message--assistant agent-chat__result";
    const label = document.createElement("span");
    label.textContent = "检测结果";
    const image = document.createElement("img");
    image.className = "agent-chat__result-image";
    image.src = `${outputPathToUrl(path)}?t=${Date.now()}`;
    image.alt = "图片异物检测带框结果";
    article.append(label, image);
    messages.appendChild(article);
    messages.scrollTop = messages.scrollHeight;
  }

  function setBusy(busy) {
    textarea.disabled = busy;
    mediaInput.disabled = busy;
    sendButton.disabled = busy;
    status.textContent = busy ? "处理中" : "待命";
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
        append(role, String(item.content || ""));
      });
    } catch (_error) {
      // History is helpful but non-blocking; chat remains usable if it fails.
    }
  }

  async function submit() {
    const message = textarea.value.trim();
    if (!message) return;
    append("user", message);
    const body = new FormData();
    body.append("message", message);
    body.append("session_id", sessionId);
    if (mediaInput.files?.[0]) body.append("media", mediaInput.files[0]);
    textarea.value = "";
    textarea.style.height = "auto";
    setBusy(true);

    try {
      const response = await fetch(endpoint, { method: "POST", body });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.error || `智能体接口返回 ${response.status}`);
      }
      append("assistant", data.reply || "操作已完成。", !data.ok);
      if (data.intent === "detect_image" && data.ok) {
        appendResultImage(data.data?.visualization_image);
      }
      if (["detect_image", "detect_video"].includes(data.intent) && data.ok) {
        mediaInput.value = "";
        fileLabel.textContent = "";
      }
    } catch (error) {
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
