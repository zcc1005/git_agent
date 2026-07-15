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
const modeTabs = document.querySelectorAll(".mode-tab");
const modePanels = document.querySelectorAll(".mode-panel");
const videoForm = document.getElementById("videoForm");
const videoInput = document.getElementById("videoInput");
const videoPreview = document.getElementById("videoPreview");
const videoFileHint = document.getElementById("videoFileHint");
const videoStartTime = document.getElementById("videoStartTime");
const videoRunButton = document.getElementById("videoRunButton");
const videoProgressFill = document.getElementById("videoProgressFill");
const videoProgressText = document.getElementById("videoProgressText");
const videoProgressPercent = document.getElementById("videoProgressPercent");
const videoEventCount = document.getElementById("videoEventCount");
const videoUniqueObjects = document.getElementById("videoUniqueObjects");
const videoPositiveFrames = document.getElementById("videoPositiveFrames");
const videoCandidateFrames = document.getElementById("videoCandidateFrames");
const videoSampledFrames = document.getElementById("videoSampledFrames");
const videoOverallRisk = document.getElementById("videoOverallRisk");
const videoClassSummary = document.getElementById("videoClassSummary");
const videoResultMessage = document.getElementById("videoResultMessage");
const videoEventRows = document.getElementById("videoEventRows");
const videoGallery = document.getElementById("videoGallery");
const videoResultPath = document.getElementById("videoResultPath");
const videoAlarmPath = document.getElementById("videoAlarmPath");
const videoAlarmReport = document.getElementById("videoAlarmReport");

let commandMode = "manual";
let progressTimer = null;
let videoProgressTimer = null;
let videoPreviewUrl = null;

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

  const riskLevelNames = { none: "无报警", low: "低风险", medium: "中风险", high: "高风险" };
  const overallRisk = data.alarm?.overall_risk || {};
  alarmSummary.textContent = data.alarm_report
    ? `规则报告已生成 · ${riskLevelNames[overallRisk.level] || overallRisk.level || "待评估"}`
    : "未生成报警报告";
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

  setBusy(true, "流程运行中");
  startSlowProgress(15, 92, "正在运行完整流程");
  commandSummary.textContent = "正在处理命令...";
  detectionSummary.textContent = "正在执行 YOLO 检测...";
  alarmSummary.textContent = "等待规则引擎评估...";

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

modeTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const selectedMode = tab.dataset.mode;
    modeTabs.forEach((item) => item.classList.toggle("active", item === tab));
    modePanels.forEach((panel) => {
      panel.classList.toggle("active", panel.id === `${selectedMode}Mode`);
    });
  });
});

function setVideoProgress(percent, text, isError = false) {
  const value = Math.max(0, Math.min(100, Math.round(percent)));
  videoProgressFill.style.width = `${value}%`;
  videoProgressFill.classList.toggle("error", isError);
  videoProgressText.textContent = text;
  videoProgressPercent.textContent = `${value}%`;
}

function stopVideoProgress() {
  if (videoProgressTimer !== null) {
    clearInterval(videoProgressTimer);
    videoProgressTimer = null;
  }
}

function startVideoProgress() {
  stopVideoProgress();
  let current = 8;
  setVideoProgress(current, "正在上传视频并逐帧检测");
  videoProgressTimer = setInterval(() => {
    current = Math.min(94, current + (current < 60 ? 2 : 1));
    setVideoProgress(current, "正在上传视频并逐帧检测");
  }, 1000);
}

function appendCell(row, text) {
  const cell = document.createElement("td");
  cell.textContent = text;
  row.appendChild(cell);
}

function updateVideoResult(data) {
  stopVideoProgress();
  setVideoProgress(100, "视频检测完成");
  videoEventCount.textContent = String(data.num_events || 0);
  videoUniqueObjects.textContent = String(data.unique_object_count || 0);
  videoPositiveFrames.textContent = String(data.positive_frames || 0);
  videoCandidateFrames.textContent = String(data.candidate_frames || 0);
  videoSampledFrames.textContent = String(data.sampled_frames || 0);
  const videoRiskNames = { none: "无报警", low: "低风险", medium: "中风险", high: "高风险" };
  videoOverallRisk.textContent = videoRiskNames[data.overall_risk?.level] || data.overall_risk?.level || "待评估";
  videoClassSummary.textContent = formatClassCounts(data.class_counts).replaceAll("，", " / ");
  videoResultPath.textContent = data.result_json || "";
  videoAlarmPath.textContent = data.alarm_report_path || "";
  videoAlarmReport.textContent = data.alarm_report || "没有生成视频报警报告";
  videoEventRows.replaceChildren();
  videoGallery.replaceChildren();

  if (!data.has_foreign_object) {
    videoResultMessage.className = "video-result-message no-detection";
    videoResultMessage.textContent = `检测完成：共按 ${data.sample_fps} FPS 检测 ${data.sampled_frames} 帧，没有确认异物。保留 ${data.candidate_frames || 0} 张候选调试帧，候选不会触发报警。`;
    const placeholder = document.createElement("div");
    placeholder.className = "gallery-placeholder";
    placeholder.textContent = "本次没有检测到异物，因此没有保存图片";
    videoGallery.appendChild(placeholder);
    return;
  }

  videoResultMessage.className = "video-result-message";
  videoResultMessage.textContent = `检测到 ${data.num_events} 个异物时间段、${data.unique_object_count} 个跟踪独立目标；共保存 ${data.saved_images} 张原始命中或候选调试帧，其中 ${data.positive_frames} 张包含确认目标。`;

  data.events.forEach((event) => {
    const row = document.createElement("tr");
    appendCell(row, String(event.event_id));
    appendCell(row, `${event.start_video_time} 至 ${event.end_video_time}`);
    appendCell(row, `${event.start_real_time} 至 ${event.end_real_time}`);
    appendCell(row, String(event.max_simultaneous_objects ?? event.object_count ?? 0));
    appendCell(row, String(event.unique_object_count ?? event.object_count ?? 0));
    appendCell(row, String(event.positive_sample_count || 0));
    appendCell(row, formatClassCounts(event.class_counts));
    appendCell(row, Number(event.max_confidence).toFixed(2));
    appendCell(row, String(event.key_frames?.length || (event.key_frame_url ? 1 : 0)));
    videoEventRows.appendChild(row);

    const keyFrames = event.key_frames?.length
      ? event.key_frames
      : [{ image_url: event.key_frame_url, track_ids: event.track_ids || [], class_counts: event.class_counts }];
    keyFrames.forEach((keyFrame, frameIndex) => {
      if (!keyFrame.image_url) return;
      const figure = document.createElement("figure");
      const image = document.createElement("img");
      image.src = `${keyFrame.image_url}?t=${Date.now()}`;
      image.alt = `异物事件 ${event.event_id} 代表帧 ${frameIndex + 1}`;
      const caption = document.createElement("figcaption");
      const trackText = keyFrame.track_ids?.length ? `轨迹 ${keyFrame.track_ids.join(", ")}` : "轨迹未记录";
      caption.textContent = `事件 ${event.event_id} · 代表帧 ${frameIndex + 1} · ${keyFrame.video_time || event.start_video_time} · ${trackText} · ${formatClassCounts(keyFrame.class_counts || event.class_counts)}`;
      figure.append(image, caption);
      videoGallery.appendChild(figure);
    });
  });
}

function showVideoError(message) {
  stopVideoProgress();
  setVideoProgress(100, "视频检测失败", true);
  videoResultMessage.className = "video-result-message no-detection";
  videoResultMessage.textContent = message;
  setStatus("出错", "error");
}

const localNow = new Date();
localNow.setMinutes(localNow.getMinutes() - localNow.getTimezoneOffset());
videoStartTime.value = localNow.toISOString().slice(0, 19);

videoInput.addEventListener("change", () => {
  const file = videoInput.files?.[0];
  if (videoPreviewUrl) {
    URL.revokeObjectURL(videoPreviewUrl);
    videoPreviewUrl = null;
  }
  if (!file) {
    videoPreview.style.display = "none";
    videoFileHint.textContent = "上传后只保存检测到异物的带框图片。";
    return;
  }
  videoPreviewUrl = URL.createObjectURL(file);
  videoPreview.src = videoPreviewUrl;
  videoPreview.style.display = "block";
  videoFileHint.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MB`;
});

videoForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!videoInput.files || videoInput.files.length === 0) {
    showVideoError("请先选择需要检测的视频。");
    return;
  }

  const body = new FormData(videoForm);
  videoRunButton.disabled = true;
  setStatus("视频检测中", "running");
  videoResultMessage.className = "video-result-message";
  videoResultMessage.textContent = "视频较长时需要等待，检测过程中页面请保持打开。";
  startVideoProgress();

  try {
    const response = await fetch("/api/video-detect", { method: "POST", body });
    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.error || "视频检测失败");
    }
    updateVideoResult(data.video_detection);
    setStatus("完成");
  } catch (error) {
    showVideoError(error.message);
  } finally {
    videoRunButton.disabled = false;
  }
});
