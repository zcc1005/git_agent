# 服务器独立上传文件清单

这些文件位于本地 `server_bundle`，必须独立于应用镜像上传到服务器。

| 服务器路径 | 本地上传文件 | 原始来源 | 用途 |
|---|---|---|---|
| `/opt/belt-agent/compose.yaml` | `server_bundle/compose.yaml` | 仓库根目录 `compose.yaml` | 编排 Web、MediaMTX 和 FFmpeg |
| `/opt/belt-agent/compose.gpu.yaml` | `server_bundle/compose.gpu.yaml` | 仓库根目录 `compose.gpu.yaml` | 为 Web 容器申请 NVIDIA GPU |
| `/opt/belt-agent/deploy/mediamtx.yml` | `server_bundle/deploy/mediamtx.yml` | `deploy/mediamtx.yml` | 注册 `main-monitor` RTSP 发布路径 |
| `/opt/belt-agent/app.env` | `server_bundle/app.env` | 由项目 `.env` 最小化生成 | 交融大模型密钥和运行配置 |
| `/opt/belt-agent/demo.mp4` | `server_bundle/demo.mp4` | `data/monitor/monitor.mp4` | FFmpeg 无限循环的模拟摄像头视频 |
| `/opt/belt-agent/belt-agent.tar` | `server_bundle/belt-agent.tar` | `docker save` 生成 | 仅离线部署需要；云效从 ACR 拉取时不需要 |

`app.env` 和 `demo.mp4` 不进入应用镜像。`app.env` 包含真实密钥，不得提交
到 Git，也不要通过聊天、工单或日志发送其内容。

使用 `yunxiao-pipeline-gpu.yaml` 部署时，流水线直接从 ACR 拉取本次镜像，
不需要上传 `belt-agent.tar`。服务器仍需提前上传其余配置、密钥和测试视频。

## 本地准备

在仓库根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\prepare_server_bundle.ps1
```

如果当前机器已安装 Docker，可同时构建并导出镜像：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\prepare_server_bundle.ps1 -BuildImage
```

如果未使用 `-BuildImage`，先单独生成镜像归档，再重新执行准备脚本：

```powershell
docker compose build web
docker save -o belt-agent.tar belt-agent:latest
powershell -ExecutionPolicy Bypass -File .\scripts\prepare_server_bundle.ps1
```

## 上传

```powershell
ssh user@server "mkdir -p /opt/belt-agent/deploy"
scp -r .\server_bundle\* user@server:/opt/belt-agent/
```

上传后在服务器限制密钥文件权限：

```bash
chmod 600 /opt/belt-agent/app.env
```

## 启动

```bash
cd /opt/belt-agent
docker load -i belt-agent.tar
APP_ENV_FILE=./app.env DEMO_VIDEO_PATH=./demo.mp4 \
  docker compose -f compose.yaml -f compose.gpu.yaml up -d --no-build
```

服务器需能拉取 `bluenviron/mediamtx:1` 和
`jrottenberg/ffmpeg:7.1-alpine`；离线服务器还需另外上传
`rtsp-images.tar`，具体方法见 `README.md`。
