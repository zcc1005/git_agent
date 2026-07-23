const navItems = document.querySelectorAll(".nav-item[data-view]");
const appViews = document.querySelectorAll(".app-view");
const statusBadge = document.getElementById("statusBadge");
const agentDrawer = document.getElementById("agentDrawer");
const agentHomeMount = document.getElementById("agentHomeMount");
const historySourceFilter = document.getElementById("historySourceFilter");
const historyRiskFilter = document.getElementById("historyRiskFilter");

const HISTORY_STORAGE_KEY = "belt-guard-front-history-v2";
const AGENT_SESSION_STORAGE_KEY = "foreign-object-agent-session";
const ALARM_RECONCILIATION_STORAGE_KEY = "belt-guard-alarm-reconciliation-v1";
const CONSOLE_SNAPSHOT_ENDPOINT = "/api/console/snapshot";
const CONSOLE_ALARM_ACTION_ENDPOINT = "/api/console/alarms/action";
const RISK_NAMES = { none: "无报警", low: "低风险", medium: "中风险", high: "高风险" };
const VIEW_NAMES = new Set(["dashboard", "alarms", "history"]);
let selectedAlarmId = "";
let alarmRiskFilter = "all";
let currentDailySummary = null;
let consoleSnapshotTimer = 0;
let consoleSnapshotPolling = false;

function setStatus(text, state = "") {
  statusBadge.replaceChildren();
  const dot = document.createElement("span");
  statusBadge.append(dot, document.createTextNode(text));
  statusBadge.className = `status-badge status-badge--sidebar ${state}`.trim();
}

function openView(viewName) {
  const view = VIEW_NAMES.has(viewName) ? viewName : "dashboard";
  navItems.forEach((item) => {
    const active = item.dataset.view === view;
    item.classList.toggle("active", active);
    if (active) item.setAttribute("aria-current", "page");
    else item.removeAttribute("aria-current");
  });
  appViews.forEach((item) => item.classList.toggle("active", item.id === `${view}View`));
  if (view === "dashboard") dockAgentHome();
  else undockAgentHome();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function dockAgentHome() {
  if (!agentHomeMount || agentDrawer.classList.contains("is-docked")) return;
  document.body.classList.remove("agent-drawer-open");
  agentDrawer.classList.add("is-docked");
  agentDrawer.setAttribute("aria-hidden", "false");
  agentHomeMount.append(agentDrawer);
}

function undockAgentHome() {
  if (!agentDrawer.classList.contains("is-docked")) return;
  agentDrawer.classList.remove("is-docked");
  agentDrawer.setAttribute("aria-hidden", "true");
  document.body.append(agentDrawer);
}

function openAgent(prompt = "") {
  if (agentDrawer.classList.contains("is-docked")) {
    if (prompt) document.dispatchEvent(new CustomEvent("agent:prefill", { detail: { prompt } }));
    else agentDrawer.querySelector("textarea[name='message']")?.focus();
    return;
  }
  document.body.classList.add("agent-drawer-open");
  agentDrawer.setAttribute("aria-hidden", "false");
  if (prompt) document.dispatchEvent(new CustomEvent("agent:prefill", { detail: { prompt } }));
  else agentDrawer.querySelector("textarea[name='message']")?.focus();
}

function closeAgent() {
  if (agentDrawer.classList.contains("is-docked")) return;
  document.body.classList.remove("agent-drawer-open");
  agentDrawer.setAttribute("aria-hidden", "true");
}

navItems.forEach((item) => item.addEventListener("click", () => openView(item.dataset.view)));

document.querySelectorAll("[data-view-target]").forEach((button) => {
  button.addEventListener("click", () => {
    openView(button.dataset.viewTarget);
    if (button.dataset.agentPrompt) openAgent(button.dataset.agentPrompt);
  });
});

document.querySelectorAll("[data-open-agent]").forEach((button) => {
  button.addEventListener("click", () => openAgent(button.dataset.agentPrompt || ""));
});

document.querySelectorAll("[data-close-agent]").forEach((button) => button.addEventListener("click", closeAgent));
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && document.body.classList.contains("agent-drawer-open")) closeAgent();
});

function setAgentTask(label, detail = "") {
  const labelNode = document.getElementById("agentPhaseLabel");
  const detailNode = document.getElementById("agentContextText");
  if (labelNode && label) labelNode.textContent = label;
  if (detailNode && detail) detailNode.textContent = detail;
}

function shortClock(value) {
  const match = String(value || "").match(/[T\s](\d{2}):(\d{2})/);
  return match ? `${match[1]}:${match[2]}` : "时间待定";
}

function realtimeTaskSentence(task) {
  const source = String(task?.display_name || task?.source_id || "监控源");
  const start = shortClock(task?.start_time || task?.started_at);
  const end = shortClock(task?.end_time || task?.stopped_at);
  const status = String(task?.status || "");
  if (["completed", "stopped", "failed", "interrupted"].includes(status)) {
    return `已结束${source}的实时巡检，时间：${start} 至 ${end}。`;
  }
  if (status === "scheduled") {
    return `已安排${source}的实时巡检，时间：${start} 至 ${end}。`;
  }
  return `正在实时巡检${source}，时间：${start} 至 ${end}。`;
}

document.addEventListener("agent:status", (event) => {
  const busy = Boolean(event.detail?.busy);
  const text = event.detail?.text || (busy ? "处理中" : "待命");
  const triggerStatus = document.getElementById("agentTriggerStatus");
  if (triggerStatus) triggerStatus.textContent = text;
  setStatus(text, busy ? "running" : "");
  if (busy) setAgentTask(text, "正在处理本次智能体任务。");
});

document.addEventListener("agent:response", (event) => {
  const response = event.detail?.data;
  if (!response || typeof response !== "object") return;
  const payload = response.data && typeof response.data === "object" ? response.data : {};
  const realtimeTask = findRealtimeTask(response);
  if (realtimeTask) {
    setAgentTask("实时监测", realtimeTaskSentence(realtimeTask));
    return;
  }
  const risk = findNestedRisk(payload);
  const detectionLike = Boolean(
    risk
    || payload.detection_id
    || payload.source_type
    || payload.event_count !== undefined
    || payload.num_events !== undefined
  );

  if (!detectionLike) {
    setAgentTask("任务已完成", "已完成本次智能体任务。");
    return;
  }

  const record = createAgentRecord(response, payload, risk);
  saveHistoryRecord(record);
  setAgentTask(
    record.riskLevel === "none" ? "分析完成" : "等待人工确认",
    `已完成一次${record.sourceType === "image" ? "图片" : record.sourceType === "video" ? "视频" : "智能体"}检测。`,
  );
});

document.addEventListener("agent:realtime-status", (event) => {
  const task = event.detail?.task;
  if (!task?.task_id) return;
  setAgentTask("实时监测", realtimeTaskSentence(task));
  const events = Array.isArray(event.detail?.events) ? event.detail.events : [];
  events.forEach((item) => saveHistoryRecord(createRealtimeEventRecord(item, task)));
});

document.addEventListener("agent:realtime-event", (event) => {
  const item = event.detail?.event;
  if (!item?.event_id) return;
  saveHistoryRecord(createRealtimeEventRecord(item, event.detail?.task || {}));
});

function normalizeRisk(value) {
  const risk = String(value || "none").toLowerCase();
  return Object.hasOwn(RISK_NAMES, risk) ? risk : "none";
}

function findNestedRisk(value) {
  if (!value || typeof value !== "object") return null;
  if (value.overall_risk && typeof value.overall_risk === "object") return value.overall_risk;
  if (value.risk_level) return { level: value.risk_level, reason: value.reason || "" };
  if (value.alarm && typeof value.alarm === "object") return findNestedRisk(value.alarm);
  if (Array.isArray(value.steps)) {
    for (let index = value.steps.length - 1; index >= 0; index -= 1) {
      const found = findNestedRisk(value.steps[index]?.data);
      if (found) return found;
    }
  }
  return null;
}

function extractReport(value) {
  if (!value || typeof value !== "object") return "";
  if (typeof value.alarm_report === "string") return value.alarm_report;
  if (typeof value.report_text === "string") return value.report_text;
  if (value.alarm) {
    const nested = extractReport(value.alarm);
    if (nested) return nested;
  }
  if (Array.isArray(value.steps)) {
    for (let index = value.steps.length - 1; index >= 0; index -= 1) {
      const nested = extractReport(value.steps[index]?.data);
      if (nested) return nested;
    }
  }
  return "";
}

function firstEventFrame(value) {
  if (!value || typeof value !== "object") return "";
  const frame = Array.isArray(value.event_frames) ? value.event_frames[0] : null;
  if (frame?.key_frame) return outputPathToUrl(frame.key_frame);
  const event = Array.isArray(value.events) ? value.events[0] : null;
  const eventFrame = event?.key_frames?.[0]?.image_url || event?.key_frame_url;
  if (eventFrame) return eventFrame;
  if (value.alarm) {
    const nested = firstEventFrame(value.alarm);
    if (nested) return nested;
  }
  if (Array.isArray(value.steps)) {
    for (let index = value.steps.length - 1; index >= 0; index -= 1) {
      const nested = firstEventFrame(value.steps[index]?.data);
      if (nested) return nested;
    }
  }
  return "";
}

function outputPathToUrl(path) {
  const normalized = String(path || "").replaceAll("\\", "/");
  if (/^https?:\/\//i.test(normalized) || normalized.startsWith("/outputs/")) return normalized;
  const marker = "/outputs/";
  const markerIndex = normalized.toLowerCase().lastIndexOf(marker);
  const relative = markerIndex >= 0
    ? normalized.slice(markerIndex + marker.length)
    : normalized.replace(/^outputs\//i, "");
  return `/outputs/${relative.split("/").map(encodeURIComponent).join("/")}`;
}

function formatClassCounts(classCounts) {
  if (!classCounts || typeof classCounts !== "object" || Object.keys(classCounts).length === 0) return "未识别具体类型";
  return Object.entries(classCounts).map(([name, count]) => `${name} ${count}`).join("，");
}

function createAgentRecord(response, payload, risk) {
  const level = normalizeRisk(risk?.level || payload.risk_level);
  const sourceType = String(payload.source_type || payload.source?.type || "agent").toLowerCase();
  const sourcePath = String(payload.source_path || payload.source?.path || "");
  const sourceName = sourcePath.split(/[\\/]/).pop() || (sourceType === "video" ? "视频巡检" : sourceType === "image" ? "图片检测" : "智能体任务");
  const eventCount = Number(payload.event_count ?? payload.num_events ?? payload.detection_count ?? 0);
  const classes = payload.class_counts || payload.detection_summary?.class_counts || {};
  const alarmStatus = String(payload.alarm_status || "").toLowerCase();
  return {
    id: String(payload.detection_id || `agent-${Date.now()}`),
    detectionId: String(payload.detection_id || ""),
    alarmId: String(payload.alarm_id || ""),
    createdAt: String(payload.created_at || new Date().toISOString()),
    sourceType: ["image", "video"].includes(sourceType) ? sourceType : "agent",
    sourceName,
    riskLevel: level,
    eventCount,
    classCounts: classes,
    classSummary: formatClassCounts(classes),
    summary: String(response.reply || `${eventCount} 个事件`),
    report: extractReport(payload) || String(response.reply || ""),
    imageUrl: firstEventFrame(payload),
    actionStatus: alarmStatus || (level === "none" ? "inactive" : "pending"),
    reason: String(risk?.reason || payload.reason || response.reply || ""),
  };
}

function realtimeEventReason(event) {
  const report = event?.alarm_report?.document || {};
  const overall = report.overall_risk || {};
  const reportEvent = Array.isArray(report.events) ? report.events[0] || {} : {};
  return String(reportEvent?.risk?.reason || overall.reason || event?.llm_summary || "实时巡检发现异物事件，等待人工复核。");
}

function createRealtimeEventRecord(event, task = {}) {
  const classCounts = event.class_counts && Object.keys(event.class_counts).length
    ? event.class_counts
    : { [String(event.class_name || "未知异物")]: 1 };
  const source = String(task.display_name || event.display_name || event.source_id || task.source_id || "监控源");
  const alarmStatus = String(event.alarm_status || "pending").toLowerCase();
  const frame = String(event.representative_frame || event.image_path || "");
  return {
    id: String(event.detection_id || `realtime-${event.task_id}-${event.event_id}`),
    detectionId: String(event.detection_id || ""),
    alarmId: String(event.alarm_id || ""),
    createdAt: String(event.detected_at || event.created_at || new Date().toISOString()),
    sourceType: "agent",
    sourceName: `${source}实时巡检`,
    riskLevel: normalizeRisk(event.risk_level),
    eventCount: 1,
    classCounts,
    classSummary: formatClassCounts(classCounts),
    summary: `实时巡检发现 ${formatClassCounts(classCounts)}`,
    report: String(event.alarm_report?.text || event.alarm_report || ""),
    imageUrl: frame ? outputPathToUrl(frame) : "",
    actionStatus: alarmStatus,
    reason: realtimeEventReason(event),
    taskId: String(event.task_id || task.task_id || ""),
    eventId: String(event.event_id || ""),
    eventStatus: String(event.event_status || "active"),
  };
}

function loadHistoryRecords() {
  try {
    const current = JSON.parse(localStorage.getItem(HISTORY_STORAGE_KEY) || "[]");
    if (Array.isArray(current)) {
      // Older builds mistakenly saved a realtime-task start response as a
      // zero-event "智能体任务".  It is not a detection or alarm record.
      return current.filter((item) => !(
        String(item?.id || "").startsWith("agent-")
        && item?.sourceName === "智能体任务"
        && Number(item?.eventCount || 0) === 0
      ));
    }
  } catch (_error) {
    // Continue with an empty local view when storage is unavailable.
  }
  return [];
}

function writeHistoryRecords(records) {
  try {
    localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(records.slice(0, 200)));
  } catch (_error) {
    // Conversation and detection remain available without local history.
  }
}

function saveHistoryRecord(record) {
  const records = loadHistoryRecords().filter((item) => item.id !== record.id);
  records.unshift(record);
  writeHistoryRecords(records);
  renderAll(records);
}

function formatRecordTime(value, includeDate = true) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat("zh-CN", {
    month: includeDate ? "2-digit" : undefined,
    day: includeDate ? "2-digit" : undefined,
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);
}

function riskClass(level) {
  return `risk-badge risk-${normalizeRisk(level)}`;
}

function actionStatusName(status) {
  return { inactive: "无需报警", pending: "待确认", confirmed: "已确认", cancelled: "已取消" }[status] || "待复核";
}

function renderDashboard(records, dailySummary = currentDailySummary) {
  const today = new Date().toDateString();
  const todayRecords = records.filter((record) => new Date(record.createdAt).toDateString() === today);
  const pending = records.filter((record) => record.riskLevel !== "none" && record.actionStatus === "pending");
  const serverRisks = dailySummary?.risk_counts || {};
  const serverStatuses = dailySummary?.alarm_status_counts || dailySummary?.status_counts || {};
  const runCount = Number(dailySummary?.detection_count ?? todayRecords.length);
  const highRiskCount = Number(serverRisks.high ?? todayRecords.filter((record) => record.riskLevel === "high").length);
  const pendingCount = Number(serverStatuses.pending ?? pending.length);
  document.getElementById("todayRunCount").textContent = String(runCount);
  document.getElementById("todayHighRiskCount").textContent = String(highRiskCount);
  document.getElementById("pendingAlarmCount").textContent = String(pendingCount);
  document.getElementById("sidebarAlarmCount").textContent = String(pendingCount);
  document.getElementById("dashboardAlertCount").textContent = `${pendingCount} 项`;
  document.getElementById("latestRiskLabel").textContent = records[0] ? RISK_NAMES[records[0].riskLevel] : "安全";

  const alertEmpty = document.getElementById("dashboardAlertEmpty");
  const alertContent = document.getElementById("dashboardAlertContent");
  alertEmpty.hidden = pending.length > 0;
  alertContent.hidden = pending.length === 0;
  if (pending.length > 0) {
    const alert = pending[0];
    const badge = document.getElementById("dashboardAlertRisk");
    badge.textContent = RISK_NAMES[alert.riskLevel];
    badge.className = riskClass(alert.riskLevel);
    document.getElementById("dashboardAlertTime").textContent = formatRecordTime(alert.createdAt, false);
    document.getElementById("dashboardAlertTitle").textContent = alert.sourceName;
    document.getElementById("dashboardAlertReason").textContent = alert.reason || alert.summary;
  }

  const list = document.getElementById("dashboardRecentList");
  list.replaceChildren();
  if (records.length === 0) {
    const empty = document.createElement("div");
    empty.className = "console-empty";
    empty.textContent = "完成对话任务后自动生成记录";
    list.append(empty);
    return;
  }
  records.slice(0, 3).forEach((record) => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "console-recent-item";
    item.addEventListener("click", () => openView("history"));
    const copy = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = record.sourceName;
    const subtitle = document.createElement("small");
    subtitle.textContent = `${RISK_NAMES[record.riskLevel]} · ${record.eventCount} 个事件`;
    copy.append(title, subtitle);
    const time = document.createElement("time");
    time.textContent = formatRecordTime(record.createdAt, false);
    item.append(copy, time);
    list.append(item);
  });
}

function alarmCenterRecords(records) {
  return records.filter((record) => (
    record.riskLevel !== "none"
    && (alarmRiskFilter === "all" || record.riskLevel === alarmRiskFilter)
  ));
}

function selectedAlarmRecord(records) {
  let filtered = alarmCenterRecords(records);
  if (!filtered.length && alarmRiskFilter !== "all") {
    alarmRiskFilter = "all";
    filtered = alarmCenterRecords(records);
  }
  let record = filtered.find((item) => item.id === selectedAlarmId) || null;
  if (!record && filtered.length) {
    record = filtered[0];
    selectedAlarmId = record.id;
  }
  return { record, filtered };
}

function renderAlarmRiskFilters(records) {
  const riskRecords = records.filter((record) => record.riskLevel !== "none");
  document.querySelectorAll("[data-alarm-risk-filter]").forEach((button) => {
    const risk = String(button.dataset.alarmRiskFilter || "all");
    const count = risk === "all"
      ? riskRecords.length
      : riskRecords.filter((record) => record.riskLevel === risk).length;
    button.classList.toggle("active", risk === alarmRiskFilter);
    const countNode = button.querySelector("span");
    if (countNode) countNode.textContent = String(count);
    button.disabled = count === 0;
  });
}

function renderAlarmCenter(records) {
  const { record, filtered } = selectedAlarmRecord(records);
  renderAlarmRiskFilters(records);
  const empty = document.getElementById("alarmCenterEmpty");
  const content = document.getElementById("alarmCenterContent");
  empty.hidden = Boolean(record);
  content.hidden = !record;
  if (!record) return;

  const position = Math.max(0, filtered.findIndex((item) => item.id === record.id));
  document.getElementById("alarmBrowserPosition").textContent = `${position + 1} / ${filtered.length}`;
  document.getElementById("previousAlarmButton").disabled = filtered.length < 2;
  document.getElementById("nextAlarmButton").disabled = filtered.length < 2;
  const pendingCount = records.filter((item) => (
    item.riskLevel !== "none" && item.actionStatus === "pending"
  )).length;
  const confirmAllButton = document.getElementById("confirmAllAlarmsButton");
  confirmAllButton.disabled = pendingCount === 0;
  confirmAllButton.textContent = pendingCount
    ? `一键确认全部待处理报警（${pendingCount}）`
    : "没有待处理报警";

  document.getElementById("alarmCenterTitle").textContent = record.sourceName;
  document.getElementById("alarmCenterSummary").textContent = record.reason || record.summary;
  const badge = document.getElementById("alarmCenterRisk");
  badge.textContent = RISK_NAMES[record.riskLevel];
  badge.className = riskClass(record.riskLevel);
  document.getElementById("alarmCenterSource").textContent = record.sourceType === "video" ? "视频巡检" : record.sourceType === "image" ? "图片检测" : "智能体任务";
  document.getElementById("alarmCenterEvents").textContent = String(record.eventCount || 0);
  document.getElementById("alarmCenterTime").textContent = formatRecordTime(record.createdAt);
  document.getElementById("alarmCenterStatus").textContent = actionStatusName(record.actionStatus);
  document.getElementById("alarmCenterReport").textContent = record.report || "暂无详细风险报告。";

  const image = document.getElementById("alarmCenterImage");
  const imageEmpty = document.getElementById("alarmCenterImageEmpty");
  if (record.imageUrl) {
    image.src = record.imageUrl;
    image.style.display = "block";
    imageEmpty.style.display = "none";
  } else {
    image.removeAttribute("src");
    image.style.display = "none";
    imageEmpty.style.display = "block";
  }
  const actionable = record.actionStatus === "pending";
  document.getElementById("confirmAlarmButton").disabled = !actionable;
  document.getElementById("cancelAlarmButton").disabled = !actionable;
}

function renderHistory(records = loadHistoryRecords()) {
  const source = historySourceFilter?.value || "all";
  const risk = historyRiskFilter?.value || "all";
  const filtered = records.filter((record) => (
    (source === "all" || record.sourceType === source)
    && (risk === "all" || record.riskLevel === risk)
  ));
  const rows = document.getElementById("historyRows");
  const empty = document.getElementById("historyEmpty");
  rows.replaceChildren();
  empty.hidden = filtered.length > 0;
  filtered.forEach((record) => {
    const row = document.createElement("tr");
    const values = [
      [formatRecordTime(record.createdAt), record.id],
      [record.sourceType === "video" ? "视频巡检" : record.sourceType === "image" ? "图片检测" : "智能体任务", record.sourceName],
      [record.summary || "检测完成", record.classSummary],
    ];
    values.forEach(([primary, secondary]) => {
      const cell = document.createElement("td");
      cell.textContent = primary;
      const small = document.createElement("small");
      small.textContent = secondary || "";
      cell.append(small);
      row.append(cell);
    });
    const riskCell = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = riskClass(record.riskLevel);
    badge.textContent = RISK_NAMES[record.riskLevel];
    riskCell.append(badge);
    const statusCell = document.createElement("td");
    statusCell.textContent = actionStatusName(record.actionStatus);
    const actionCell = document.createElement("td");
    const action = document.createElement("button");
    action.type = "button";
    action.className = "text-button";
    action.textContent = "智能体复核";
    action.addEventListener("click", () => openAgent(`请查询并复核检测记录 ${record.id}`));
    actionCell.append(action);
    row.append(riskCell, statusCell, actionCell);
    rows.append(row);
  });
}

function renderAll(records = loadHistoryRecords()) {
  renderDashboard(records, currentDailySummary);
  renderAlarmCenter(records);
  renderHistory(records);
}

function applyConsoleSnapshot(snapshot) {
  if (!snapshot || typeof snapshot !== "object") return;
  currentDailySummary = snapshot.summary && typeof snapshot.summary === "object"
    ? snapshot.summary
    : currentDailySummary;
  const localRecords = loadHistoryRecords();
  const localById = new Map(localRecords.map((record) => [String(record.id), record]));
  const serverRecords = Array.isArray(snapshot.records) ? snapshot.records : [];
  const merged = serverRecords.map((serverRecord) => {
    const local = localById.get(String(serverRecord.id)) || {};
    localById.delete(String(serverRecord.id));
    const classCounts = serverRecord.classCounts || local.classCounts || {};
    return {
      ...local,
      ...serverRecord,
      sourceName: local.sourceName && local.sourceName !== "智能体任务"
        ? local.sourceName
        : serverRecord.sourceName,
      classCounts,
      classSummary: formatClassCounts(classCounts),
      imageUrl: serverRecord.imageUrl || local.imageUrl || "",
      report: serverRecord.report || local.report || "",
      reason: serverRecord.reason || local.reason || "",
    };
  });
  localById.forEach((record) => merged.push(record));
  merged.sort((left, right) => String(right.createdAt || "").localeCompare(String(left.createdAt || "")));
  writeHistoryRecords(merged);
  renderAll(merged);
}

function scheduleConsoleSnapshot(delay = 3000) {
  window.clearTimeout(consoleSnapshotTimer);
  consoleSnapshotTimer = window.setTimeout(() => refreshConsoleSnapshot(), delay);
}

async function refreshConsoleSnapshot() {
  if (consoleSnapshotPolling) {
    scheduleConsoleSnapshot(250);
    return;
  }
  consoleSnapshotPolling = true;
  try {
    const response = await fetch(`${CONSOLE_SNAPSHOT_ENDPOINT}?limit=200`, { cache: "no-store" });
    const data = await response.json().catch(() => ({}));
    if (response.ok && data.ok) applyConsoleSnapshot(data);
  } catch (_error) {
    // Local records remain usable while the authoritative snapshot is temporarily unavailable.
  } finally {
    consoleSnapshotPolling = false;
    scheduleConsoleSnapshot(3000);
  }
}

async function reconcileLegacyAlarmActions() {
  if (localStorage.getItem(ALARM_RECONCILIATION_STORAGE_KEY) === "done") return;
  const records = loadHistoryRecords();
  const groups = {
    confirm: records.filter((record) => record.actionStatus === "confirmed"),
    cancel: records.filter((record) => record.actionStatus === "cancelled"),
  };
  try {
    for (const [action, targets] of Object.entries(groups)) {
      if (!targets.length) continue;
      const response = await fetch(CONSOLE_ALARM_ACTION_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action,
          reconcile_only: true,
          session_id: localStorage.getItem(AGENT_SESSION_STORAGE_KEY) || "default",
          detection_ids: targets.map((record) => record.detectionId || record.id).filter(Boolean),
          note: "迁移旧版网页中已经完成的人工报警处置状态",
        }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data.ok) throw new Error(data.error || "旧状态同步失败");
    }
    localStorage.setItem(ALARM_RECONCILIATION_STORAGE_KEY, "done");
  } catch (_error) {
    // Keep the marker unset so a later page load can retry the one-time migration.
  }
}

async function sendAlarmAction(action, { allPending = false } = {}) {
  const records = loadHistoryRecords();
  const selected = records.find((record) => record.id === selectedAlarmId);
  const targets = allPending
    ? records.filter((record) => record.riskLevel !== "none" && record.actionStatus === "pending")
    : selected && selected.actionStatus === "pending" ? [selected] : [];
  if (!targets.length) return;
  const actionName = action === "confirm" ? "确认报警" : "取消报警";
  setStatus(allPending ? `${actionName}（${targets.length}条）` : actionName, "running");
  try {
    const response = await fetch(CONSOLE_ALARM_ACTION_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        action,
        all_pending: allPending,
        session_id: localStorage.getItem(AGENT_SESSION_STORAGE_KEY) || "default",
        alarm_ids: targets.map((record) => record.alarmId).filter(Boolean),
        detection_ids: targets.map((record) => record.detectionId || record.id).filter(Boolean),
        note: allPending ? "网页报警中心一键处置全部待处理报警" : "网页报警中心单条处置",
      }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) throw new Error(data.error || "报警控制失败");
    if (data.snapshot) applyConsoleSnapshot(data.snapshot);
    else scheduleConsoleSnapshot(0);
    const affected = Number(data.affected_count || 0);
    setStatus(action === "confirm" ? "已确认" : "已取消");
    setAgentTask(
      action === "confirm" ? "报警已确认" : "报警已取消",
      `本次已处置 ${affected} 条报警，SQLite 与看板状态已同步。`,
    );
  } catch (error) {
    setStatus("处置失败", "error");
    setAgentTask("处置失败", "报警处置未完成，请在聊天框中重试。");
  }
}

document.getElementById("confirmAlarmButton")?.addEventListener("click", () => {
  if (window.confirm("确认继续当前报警？该操作将同步写入 SQLite 报警记录。")) {
    sendAlarmAction("confirm");
  }
});

document.getElementById("cancelAlarmButton")?.addEventListener("click", () => {
  if (window.confirm("确认取消当前报警？请仅在误报或风险已经解除时操作。")) {
    sendAlarmAction("cancel");
  }
});

document.getElementById("confirmAllAlarmsButton")?.addEventListener("click", () => {
  const pendingCount = loadHistoryRecords().filter((record) => (
    record.riskLevel !== "none" && record.actionStatus === "pending"
  )).length;
  if (pendingCount && window.confirm(`确认一次性继续全部 ${pendingCount} 条待处理报警？`)) {
    sendAlarmAction("confirm", { allPending: true });
  }
});

function moveSelectedAlarm(offset) {
  const records = loadHistoryRecords();
  const { record, filtered } = selectedAlarmRecord(records);
  if (!record || filtered.length < 2) return;
  const currentIndex = filtered.findIndex((item) => item.id === record.id);
  const nextIndex = (currentIndex + offset + filtered.length) % filtered.length;
  selectedAlarmId = filtered[nextIndex].id;
  renderAlarmCenter(records);
}

document.getElementById("previousAlarmButton")?.addEventListener("click", () => moveSelectedAlarm(-1));
document.getElementById("nextAlarmButton")?.addEventListener("click", () => moveSelectedAlarm(1));
document.querySelectorAll("[data-alarm-risk-filter]").forEach((button) => {
  button.addEventListener("click", () => {
    alarmRiskFilter = String(button.dataset.alarmRiskFilter || "all");
    selectedAlarmId = "";
    renderAlarmCenter(loadHistoryRecords());
  });
});

historySourceFilter?.addEventListener("change", () => renderHistory());
historyRiskFilter?.addEventListener("change", () => renderHistory());

dockAgentHome();
renderAll();
reconcileLegacyAlarmActions().finally(() => refreshConsoleSnapshot());
