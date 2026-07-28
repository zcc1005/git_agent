# 服务器部署说明

## 1. 构建与传输镜像

在仓库根目录构建。Dockerfile 只打包一份生产权重
`runs/yolo/yiwu_yolov8s_4class/weights/best.pt`，容器内路径为
`/models/best.pt`，不会打包其他训练 checkpoint。

当前生产 `best.pt` 已被仓库跟踪；但 `.gitignore` 会拦截以后新增的其他权重。
执行构建的机器仍应先确认该文件存在。若权重缺失，Docker 构建会直接失败，
避免生成一个“能启动但不能检测”的残缺镜像。

```powershell
docker compose build
docker save -o belt-agent.tar belt-agent:latest
```

在服务器导入：

```bash
docker load -i belt-agent.tar
docker run --rm --entrypoint sh belt-agent:latest \
  -c 'test -s /models/best.pt && ls -lh /models/best.pt'
```

## 2. API 密钥

API 密钥不应写入镜像，否则任何能拉取镜像的人都能读取密钥。服务器必须
另外准备项目 `.env`，可以复制 `.env.example` 后填写：

```env
LLM_PROVIDER=c4ai
LLM_C4AI_API_KEY=真实密钥
LLM_C4AI_BASE_URL=https://c4ai.ccccltd.cn/api/compatible/v1
LLM_C4AI_MODEL=jiaorong-deepseek-v4-pro
```

使用 `docker compose up -d` 时，`compose.yaml` 会把该文件注入容器环境。
服务器只复制了镜像和 `compose.yaml` 时，可以把密钥写入 `/opt/belt-agent/app.env`
并这样启动：

```bash
cd /opt/belt-agent
APP_ENV_FILE=./app.env docker compose up -d --no-build
```

如果只使用 `docker run`：

```bash
docker run -d --name belt-agent \
  --env-file /opt/belt-agent/.env \
  -e YOLO_MODEL_PATH=/models/best.pt \
  -p 8000:5000 \
  -v belt_agent_outputs:/app/outputs \
  --add-host=host.docker.internal:host-gateway \
  belt-agent:latest
```

## 3. 图片、视频和检测结果

上传原文件、浏览器兼容的视频预览、结果图片、检测 JSON、告警报告和聊天
历史都保存在 `/app/outputs`。它们不需要预先打进镜像，必须使用持久卷：

```yaml
volumes:
  - app_outputs:/app/outputs
```

Web 服务通过 `/outputs/<文件路径>` 返回这些媒体。若服务器前面使用 Nginx，
需要把 `/outputs/` 和 `/api/` 一并反向代理到应用，且不要把
`/outputs/` 指向宿主机的本地静态目录。上传大视频时还应设置：

```nginx
client_max_body_size 2g;
proxy_read_timeout 3600s;
```

## 4. Compose 虚拟 RTSP 摄像机

Compose 默认启动三个服务：

- `rtsp`：MediaMTX，只提供 RTSP over TCP。
- `rtsp-publisher`：FFmpeg 按原始速度无限循环播放 `/media/demo.mp4`，并转为
  H.264 推流；每秒设置一个关键帧，方便检测程序快速连接并读取首帧。
- `web`：通过 `rtsp://rtsp:8554/main-monitor` 拉流、采集和检测。

服务器创建部署目录并上传文件：

```bash
mkdir -p /opt/belt-agent/deploy
```

需要放入：

```text
/opt/belt-agent/compose.yaml
/opt/belt-agent/deploy/mediamtx.yml
/opt/belt-agent/app.env
/opt/belt-agent/demo.mp4
/opt/belt-agent/belt-agent.tar
```

本项目现有的 `data/monitor/monitor.mp4` 约 100 秒、30 FPS，可直接上传并在
服务器命名为 `demo.mp4`。如果从开发机复制：

```powershell
scp .\deep_learning_practice_tasks1_2\deep_learning_practice_project\data\monitor\monitor.mp4 `
  user@server:/opt/belt-agent/demo.mp4
scp .\compose.yaml user@server:/opt/belt-agent/compose.yaml
scp .\deploy\mediamtx.yml user@server:/opt/belt-agent/deploy/mediamtx.yml
scp .\belt-agent.tar user@server:/opt/belt-agent/belt-agent.tar
```

服务器启动：

```bash
cd /opt/belt-agent
docker load -i belt-agent.tar
APP_ENV_FILE=./app.env DEMO_VIDEO_PATH=./demo.mp4 \
  docker compose up -d --no-build
```

服务器需要能从镜像仓库拉取 `bluenviron/mediamtx:1` 和
`jrottenberg/ffmpeg:7.1-alpine`。如果服务器不能联网，可在联网机器先执行：

```bash
docker pull bluenviron/mediamtx:1
docker pull jrottenberg/ffmpeg:7.1-alpine
docker save -o rtsp-images.tar \
  bluenviron/mediamtx:1 jrottenberg/ffmpeg:7.1-alpine
```

把 `rtsp-images.tar` 传到服务器并执行 `docker load -i rtsp-images.tar`。

`main-monitor` 的注册地址保持不变，智能体只会把它作为普通 RTSP 摄像头。
Compose 内部地址不需要开放到公网；宿主机的 8554 只绑定在
`127.0.0.1`，可供服务器本机验证。

查看三个服务状态：

```bash
docker compose ps
docker compose logs -f rtsp rtsp-publisher web
```

在 Web 容器内验证拉流和解码：

```bash
docker compose exec web sh -c \
  'ffprobe -rtsp_transport tcp -v error -show_streams "$MAIN_MONITOR_RTSP_URL"'
```

也可以从服务器宿主机验证：

```bash
ffprobe -rtsp_transport tcp \
  -v error -show_streams rtsp://127.0.0.1:8554/main-monitor
```

启动后在对话框依次测试：

1. `检查主监控是否在线`
2. `检测主监控最近 60 秒的视频`
3. `启动主监控实时巡检 10 分钟`

第 2 条会把 60 秒采集片段写入 `/app/outputs/rtsp_captures`，随后执行 YOLO
检测并生成结果图片、浏览器预览、JSON 和告警。第 3 条按视频源配置中的
`segment_seconds: 60` 反复采集和逐段检测，不会一次把十分钟视频读进内存。

## 5. 部署自检

```bash
curl http://127.0.0.1:8000/api/health
docker compose logs -f web
```

健康接口只报告权重、LLM 配置和媒体目录是否就绪，不返回 API 密钥或 RTSP
凭据。

## 6. NVIDIA GPU 与云效流水线

GPU 镜像使用根目录的 `Dockerfile.gpu`，基础镜像为：

```text
pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
```

图片与视频检测通过 `YOLO_DEVICE=0` 显式选择第一张 GPU。服务器还必须安装
NVIDIA 驱动和 NVIDIA Container Toolkit；仅有宿主机 `nvidia-smi` 不代表
Docker 容器一定能访问 GPU。

云效可直接使用根目录 `yunxiao-pipeline-gpu.yaml`。它会：

1. 使用 `Dockerfile.gpu` 构建并推送 CUDA 镜像到 ACR。
2. 在更新服务前，用本次镜像实际执行一次 64×64 YOLO GPU 推理。
3. 从镜像自动提取 Compose、MediaMTX 配置和 demo 视频到 `/opt/belt-agent`。
4. 使用 `compose.yaml + compose.gpu.yaml` 启动 Web、MediaMTX 和 FFmpeg。
5. 强制检查运行容器中的 `torch.cuda.is_available()` 和 `YOLO_DEVICE=0`。
6. 任一 GPU 检查失败时终止部署，不再静默回退到 CPU。

云效部署直接从 ACR 拉镜像，因此不需要 `belt-agent.tar`，也无需手工上传
Compose、MediaMTX 配置和 demo 视频。`app.env` 有两种准备方式：

1. 提前上传 `/opt/belt-agent/app.env`，流水线会一直复用。
2. 在云效中添加保密变量 `LLM_C4AI_API_KEY`，首次部署时自动生成。

不要把 `LLM_C4AI_API_KEY` 配置成普通字符串变量，也不要写入流水线 YAML。

服务器最低自检：

```bash
nvidia-smi
docker compose version
docker run --rm --gpus all \
  nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

CUDA 12.1 镜像体积显著大于 CPU 镜像，官方 PyTorch 基础层约 3 GB；应确认
ACR 容量、服务器磁盘空间和首次拉取超时时间充足。
