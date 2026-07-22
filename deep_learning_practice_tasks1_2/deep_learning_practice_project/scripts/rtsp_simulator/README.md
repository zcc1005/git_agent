# 本地 RTSP 模拟器

本目录用于第一阶段 RTSP 基础设施验证：把本地 MP4 按真实速度循环推送到本机 MediaMTX，并从 RTSP 地址抓取验证帧。它不会调用 YOLO，也不会修改现有视频检测逻辑。

## 准备依赖

项目需要：

- `.local/rtsp_simulator/bin/ffmpeg.exe`
- `.local/rtsp_simulator/mediamtx/mediamtx.exe`

确认依赖：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\rtsp_simulator\setup.ps1 `
  -PythonPath "C:\Users\你的用户名\anaconda3\envs\dl_practice\python.exe"
```

FFmpeg 可由项目 Python 环境中的 `imageio-ffmpeg` 提供。MediaMTX 需要从官方 Release 下载 Windows amd64 压缩包，解压到上面的目录。`.local/` 已被 Git 忽略。

## 生成测试视频

生成20秒测试画面：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\rtsp_simulator\create_sample_video.ps1
```

输出文件为 `outputs/rtsp_simulator/sample.mp4`。也可以跳过这一步，直接使用自己的 MP4。

## 启动循环推流

使用测试视频：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\rtsp_simulator\start.ps1 `
  -VideoPath .\outputs\rtsp_simulator\sample.mp4 `
  -StreamName main-monitor
```

使用自己的视频：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\rtsp_simulator\start.ps1 `
  -VideoPath "D:\videos\belt.mp4" `
  -StreamName main-monitor
```

默认 RTSP 地址：

```text
rtsp://127.0.0.1:8554/main-monitor
```

MediaMTX 和 FFmpeg 会作为隐藏后台进程运行。PID 位于 `.local/rtsp_simulator/runtime/`，MediaMTX 日志位于 `.local/rtsp_simulator/mediamtx/mediamtx.log`。

## 验证视频流

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\rtsp_simulator\verify.ps1 `
  -StreamName main-monitor
```

成功时返回 `ok=true`，并从 RTSP 流抓取一帧保存到：

```text
outputs/rtsp_simulator/verified-frame.jpg
```

也可以用 VLC 打开 `rtsp://127.0.0.1:8554/main-monitor`，直观看到循环播放画面。

## 停止模拟器

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\rtsp_simulator\stop.ps1 `
  -StreamName main-monitor
```

停止脚本只会结束 PID 文件记录且可执行文件路径核验一致的进程。

## 验收标准

1. `start.ps1` 返回 `ok=true` 和 RTSP 地址。
2. `verify.ps1` 返回 `ok=true` 并生成非空验证帧。
3. 视频原始时长结束后再次运行 `verify.ps1` 仍能抓帧，证明循环推流有效。
4. `stop.ps1` 返回两个进程均已停止。
5. 停止后本机端口 `8554` 不再监听。

## 常见问题

- 端口8554被占用：先停止已有 RTSP 服务或上一次模拟器任务。
- 推流启动失败：查看 `mediamtx.log`，确认出现 `stream is available and online`。
- 验证失败：确认 MediaMTX 与 FFmpeg 两个 PID 仍存在，再检查 Windows 防火墙或视频编码。
- 更换视频：先运行 `stop.ps1`，再用新的 `-VideoPath` 启动。
