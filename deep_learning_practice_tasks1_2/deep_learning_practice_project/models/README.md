# Deployment Models

本目录只放置运行演示所需的最终模型，不保存训练过程、优化器状态或中间 checkpoint。

目录约定：

```text
models/
  yolo/
    best.pt
```

模型二进制默认不提交 Git。复制模型时应同时记录来源训练目录、验证指标、文件大小和 SHA256。

当前生产默认仍使用 `runs/yolo/yiwu_yolov8s_4class/weights/best.pt`；在统一配置阶段完成验证后，再将其复制为演示模型并切换配置。
