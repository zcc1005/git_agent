const form = document.getElementById("pipelineForm");
const micButton = document.getElementById("micButton");
const runButton = document.getElementById("runButton");
const statusBadge = document.getElementById("statusBadge");
const recordSeconds = document.getElementById("recordSeconds");
const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const micResult = document.getElementById("micResult");
const commandSummary = document.getElementById("commandSummary");
const detectionSummary = document.getElementById("detectionSummary");
const alarmSummary = document.getElementById("alarmSummary");
const commandJson = document.getElementById("commandJson");
const detectionJson = document.getElementById("detectionJson");
const alarmReport = document.getElementById("alarmReport");
const resultImage = document.getElementById("resultImage");
const imagePlaceholder = document.getElementById("imagePlaceholder");
const imagePath = document.getElementById("imagePath");
const alarmPath = document.getElementById("alarmPath");
const progressFill = document.getElementById("progressFill");
const progressText = document.getElementById("progressText");
const progressPercent = document.getElementById("progressPercent");

let commandMode = "manual";
let progressTimer = null;

function setProgress(percent, text, isError = false) {
  const value = Math.max(0, Math.min(100, Math.round(percent)));
  progressFill.style.width = `${value}%`;
  progressFill.classList.toggle("error", isError);
  progressText.textContent = text;
  progressPercent.textContent = `${value}%`;
}

function stopProgressTimer() {
  if (progressTimer !== null) {
    clearInterval(progressTimer);
    progressTimer = null;
  }
}

function startSlowProgress(start, limit, text) {
  stopProgressTimer();
  setProgress(start, text);
  let current = start;
  progressTimer = setInterval(() => {
    if (current < limit) {
      current += current < 60 ? 2 : 1;
      setProgress(Math.min(current, limit), text);
    }
  }, 900);
}

function setStatus(text, state = "") {
  statusBadge.textContent = text;
  statusBadge.className = `status-badge ${state}`.trim();
}

function setBusy(isBusy, text = "运行中") {
  micButton.disabled = isBusy;
  runButton.disabled = isBusy;
  document.querySelectorAll("input[name='command']").forEach((input) => {
    input.disabled = isBusy;
  });
  setStatus(isBusy ? text : "就绪", isBusy ? "running" : "");
}

function showError(message) {
  stopProgressTimer();
  setProgress(100, "运行失败", true);
  setStatus("出错", "error");
  alarmSummary.textContent = "流程失败";
  alarmReport.textContent = message;
}

function prettyJson(data) {
  return JSON.stringify(data || {}, null, 2);
}

function formatClassCounts(classCounts) {
  if (!classCounts || Object.keys(classCounts).length === 0) {
    return "未检测到具体类型";
  }
  return Object.entries(classCounts)
    .map(([name, count]) => `${name} ${count}`)
    .join("，");
}

function updateCommand(command) {
  const cmd = command?.command || "未知";
  const meaning = command?.meaning || "";
  const confidence = command?.confidence ?? "";
  commandSummary.textContent = confidence === ""
    ? `${cmd} / ${meaning}`
    : `${cmd} / ${meaning} / 置信度 ${confidence}`;
  commandJson.textContent = prettyJson(command);
}

function updateResult(data) {
  stopProgressTimer();
  setProgress(100, "流程完成");
  updateCommand(data.command);

  const detection = data.detection || {};
  const hasForeignObject = detection.has_foreign_object ?? detection.has_yiwu;
  const classSummary = formatClassCounts(detection.class_counts);
  detectionSummary.textContent = `${detection.num_images || 0} 张图片，${detection.num_detections || 0} 个目标，异物：${Boolean(hasForeignObject)}，类型：${classSummary}`;
  detectionJson.textContent = prettyJson(detection);

  alarmSummary.textContent = data.alarm_report ? "报警报告已生成" : "未生成报警报告";
  alarmReport.textContent = data.alarm_report || "无报警报告内容";
  alarmPath.textContent = data.paths?.alarm_report || "";
  imagePath.textContent = data.paths?.visualization_image || "";

  if (data.image_url) {
    resultImage.src = `${data.image_url}?t=${Date.now()}`;
    resultImage.style.display = "block";
    imagePlaceholder.style.display = "none";
  } else {
    resultImage.removeAttribute("src");
    resultImage.style.display = "none";
    imagePlaceholder.style.display = "block";
    imagePlaceholder.textContent = "当前命令未启动检测，或没有生成可视化图片";
  }
}

async function sendAlarmAction(action) {
  const body = new FormData();
  body.append("action", action);

  setBusy(true, action === "yes" ? "继续报警" : "停止报警");
  setProgress(action === "yes" ? 85 : 80, action === "yes" ? "正在确认继续报警" : "正在停止报警");

  try {
    const response = await fetch("/api/alarm_action", { method: "POST", body });
    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.error || "报警控制失败");
    }
    updateResult(data);
    micResult.textContent = data.message;
    alarmSummary.textContent = data.message;
    setStatus(action === "yes" ? "继续报警" : "已停止");
  } catch (error) {
    showError(error.message);
  } finally {
    micButton.disabled = false;
    runButton.disabled = false;
    document.querySelectorAll("input[name='command']").forEach((input) => {
      input.disabled = false;
    });
  }
}

document.querySelectorAll("input[name='command']").forEach((input) => {
  input.addEventListener("change", () => {
    const value = input.value;
    commandMode = "manual";
    if (value === "yes" || value === "no") {
      micResult.textContent = value === "yes" ? "正在确认并继续当前报警。" : "正在停止当前报警。";
      sendAlarmAction(value);
    } else {
      micResult.textContent = "当前使用手动命令。";
    }
  });
});

imageInput.addEventListener("change", () => {
  const file = imageInput.files?.[0];
  if (!file) {
    previewImage.style.display = "none";
    return;
  }
  previewImage.src = URL.createObjectURL(file);
  previewImage.style.display = "block";
});

async function runPipeline() {
  if (!imageInput.files || imageInput.files.length === 0) {
    showError("请先选择需要检测的图片。");
    return;
  }

  const body = new FormData();
  body.append("mode", commandMode);
  body.append("command", document.querySelector("input[name='command']:checked").value);
  body.append("image", imageInput.files[0]);
  body.append("conf", document.getElementById("confInput").value || "0.15");
  body.append("top_k", document.getElementById("topKInput").value || "5");
  body.append("qwen_device", document.getElementById("qwenDevice").value || "cpu");

  setBusy(true, "流程运行中");
  startSlowProgress(15, 92, "正在运行完整流程");
  commandSummary.textContent = "正在处理命令...";
  detectionSummary.textContent = "正在执行 YOLO 检测...";
  alarmSummary.textContent = "等待 LoRA-Qwen 生成...";

  try {
    const response = await fetch("/api/run", { method: "POST", body });
    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.error || "流程运行失败");
    }
    updateResult(data);
    setStatus("完成");
  } catch (error) {
    showError(error.message);
  } finally {
    micButton.disabled = false;
    runButton.disabled = false;
    document.querySelectorAll("input[name='command']").forEach((input) => {
      input.disabled = false;
    });
  }
}

micButton.addEventListener("click", async () => {
  const body = new FormData();
  body.append("record_seconds", recordSeconds.value || "2.5");
  body.append("sample_rate", "16000");
  let continuedAfterMic = false;

  setBusy(true, "录音中");
  startSlowProgress(10, 45, "正在录音并识别语音命令");
  micResult.textContent = `请在 ${recordSeconds.value || "2.5"} 秒内说出 go / stop / yes / no。`;

  try {
    const response = await fetch("/api/mic", { method: "POST", body });
    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.error || "语音识别失败");
    }
    commandMode = "mic";
    stopProgressTimer();
    setProgress(50, "语音命令识别完成");
    updateCommand(data.command);
    micResult.textContent = `语音识别完成：${data.command.command} / ${data.command.meaning}，录音保存为 ${data.wav_path}`;
    setStatus("语音完成");

    const commandValue = data.command.command;
    if (commandValue === "yes" || commandValue === "no") {
      continuedAfterMic = true;
      await sendAlarmAction(commandValue);
      return;
    }
    if (imageInput.files && imageInput.files.length > 0) {
      micResult.textContent += "，已自动进入检测流程。";
      continuedAfterMic = true;
      await runPipeline();
      return;
    }
    micResult.textContent += "。请先选择图片，再点击运行完整流程。";
  } catch (error) {
    showError(error.message);
  } finally {
    micButton.disabled = false;
    runButton.disabled = false;
    document.querySelectorAll("input[name='command']").forEach((input) => {
      input.disabled = false;
    });
    if (!continuedAfterMic && !statusBadge.classList.contains("error")) {
      setStatus("就绪");
    }
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await runPipeline();
});
