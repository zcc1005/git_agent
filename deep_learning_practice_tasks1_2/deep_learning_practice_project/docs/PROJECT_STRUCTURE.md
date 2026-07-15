# 项目目录规范

```text
deep_learning_practice_project/
  config/             可提交的默认配置与演示配置
  data/
    demo/             最小可复现演示数据
    yolo_yiwu/        正式 YOLO 数据集
    yolo_unknown_eval/未知类评测数据
  docs/               项目结构与设计文档
  models/             部署所需最终模型，不放训练中间产物
  outputs/            上传文件、检测 JSON、框图和报警报告
  runs/               训练过程、曲线、日志和 checkpoint
  scripts/            启动自检和演示辅助脚本
  static/             Web 静态资源
  templates/          Web 页面模板
  task2_yolo/         YOLO 数据、训练和图片检测能力
  task3_alarm/        统一报警结构与规则引擎
  tests/              可重复运行的回归测试
  project_config.py   当前统一路径入口
  main_pipeline.py    图片完整流程入口
  video_detection.py 视频检测与事件聚合
  web_app.py          Web 服务入口
```

## 边界规则

- `data/` 是输入数据；`outputs/` 是运行结果；`runs/` 是训练产物，三者不得混用。
- `models/` 只保存已选定的部署权重，训练产生的 `best.pt` 和 `last.pt` 仍归属于 `runs/`。
- 核心业务逻辑放在任务模块中，项目级检查和维护命令放在 `scripts/`。
- 测试统一放在 `tests/`，测试生成的临时文件不得写入正式 `outputs/`。
- 配置中不写个人电脑绝对路径，本机差异通过环境变量或未提交的 `config/local.yaml` 处理。

## 当前迁移状态

本次规范化只建立目录边界并归拢测试，没有移动现有数据、模型权重、训练结果或检测输出。后续按以下顺序迁移：

1. 建立 `config/default.yaml` 并接入 `project_config.py`。
2. 复制已验证的最佳权重到 `models/yolo/best.pt`，校验 SHA256 后切换演示配置。
3. 选择一张图片和一段短视频放入 `data/demo/`。
4. 在 `scripts/` 实现启动前自检和固定演示验证。
