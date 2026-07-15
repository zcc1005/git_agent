# Scripts

本目录存放项目级运维脚本，不存放核心检测业务逻辑。

计划加入：

- `preflight_check.py`：启动前检查依赖、模型、示例数据、输出目录和端口。
- `run_demo.py`：按固定样例运行完整图片和视频演示。
- `verify_demo_outputs.py`：验证演示输出关键字段。

现有训练和检测脚本继续保留在各自的 `task2_yolo/` 和 `task3_alarm/` 中。
