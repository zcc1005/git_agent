# 智能体 Web 接入说明

智能体已接入 `web_app.py`：意图识别、工具路由、SQLite 历史和业务回复由进程级 `agent.AgentService` 负责，Web 层提供下面两个接口。

也可以通过命令行直接验证：

```powershell
python -m agent "今天有几次高风险报警"
python -m agent "检测这段视频" --video data/demo/example.mp4
```

## 0. 预留的大模型混合模式

`AgentService` 默认使用 `hybrid`，目前未注入模型时会自动退化为纯规则模式，不产生网络请求，也不需要新增依赖。

| 模式 | 行为 |
| --- | --- |
| `rules` | 只运行确定性规则 |
| `hybrid` | 规则优先；规则无法判断时才调用模型识别器 |
| `model` | 所有消息调用模型，但报警控制仍受显式指令安全门约束 |

未来只需实现 `IntentRecognizer` 协议并注入，无需修改工具路由：

```python
from agent import AgentService, Intent, IntentMatch


class FutureModelIntentRecognizer:
    def recognize(self, text: str, *, context=None) -> IntentMatch:
        # 后续在这里调用实际大模型，并把结构化结果校验为封闭的 Intent 枚举。
        return IntentMatch(
            intent=Intent.GENERATE_DAILY_REPORT,
            confidence=0.91,
            slots={},
            source="model",
        )


agent_service = AgentService(
    store,
    model_recognizer=FutureModelIntentRecognizer(),
    recognition_mode="hybrid",
    model_confidence_threshold=0.75,
)
```

传给模型识别器的 `context` 包含：

- `session_id`：当前会话。
- `history`：最近 12 条 SQLite 会话消息。
- `request_context`：本轮视频路径、视频开始时间等非二进制上下文。

模型只能提出 `Intent + confidence + slots`，不能返回或执行任意函数名。最终仍由 `ToolRouter` 将封闭意图映射到现有业务工具。模型推断出的 `confirm_alarm` / `cancel_alarm` 如果没有同时命中显式规则，会返回 `unknown`，不会执行报警控制。

## 1. `POST /api/agent/chat`

请求使用 `multipart/form-data`：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `message` | 是 | 自然语言消息 |
| `session_id` | 否 | 会话 ID，缺省为 `default` |
| `media` | 否 | “检测这张图片”或“检测这段视频”时上传的图片/视频 |
| `video_start_time` | 否 | ISO 8601 时间，缺省为服务器当前时间 |

当前 Web 适配流程：

1. 如果存在 `media`，按扩展名保存为图片或视频；接口仍兼容旧的 `image` / `video` 字段。
2. 构造 `context={"image_path": ...}` 或 `context={"video_path": ..., "video_start_time": ...}`。
3. 调用 `AgentService.chat(message, session_id=session_id, context=context)`。
4. 原样 JSON 返回服务结果。视频检测耗时较长时，接口可在后续替换为任务队列；`AgentTools` 已支持注入 detection runner。

响应示例：

```json
{
  "ok": true,
  "session_id": "d7dd...",
  "intent": "count_high_risk_today",
  "confidence": 1.0,
  "tool_name": "high_risk_counter",
  "reply": "2026-07-16 共记录 2 次高风险报警。",
  "data": {
    "date": "2026-07-16",
    "high_risk_count": 2
  }
}
```

## 2. `GET /api/agent/history`

查询参数：

- `session_id`：缺省为 `default`。
- `limit`：缺省为 `50`，建议限制为 `1..200`。

返回：

```json
{
  "ok": true,
  "messages": []
}
```

数据来自 `AgentService.history(session_id, limit)`。

## 3. 前端组件挂载

当前页面已完成以下挂载：

```jinja2
<link rel="stylesheet" href="{{ url_for('static', filename='agent_chat/agent_chat.css') }}">
{% include "components/agent_chat.html" %}
<script type="module" src="{{ url_for('static', filename='agent_chat/agent_chat.js') }}"></script>
```

组件包含聊天输入框、视频附件、快捷指令、SQLite 历史恢复和消息区，不依赖现有 `web_app.js`。

## 4. 报警控制的设备侧回调

SQLite 会记录报警的 `pending / confirmed / cancelled` 状态。若确认/取消还需要驱动声光报警或 PLC，在创建 `AgentTools` 时传入：

```python
AgentTools(store, alarm_control_handler=my_alarm_control)
```

回调签名为 `handler(action, alarm_record)`，其中 `action` 为 `confirm` 或 `cancel`。回调成功后才更新 SQLite；回调异常时状态保持不变，便于重试和审计。

独立使用 Agent 时，如需沿用当前 `web_app.py` 中恢复/停止活动报告的完整逻辑，可使用延迟导入适配器：

```python
from agent import AgentService, AgentTools, existing_web_alarm_control

tools = AgentTools(store, alarm_control_handler=existing_web_alarm_control)
agent_service = AgentService(store, tools=tools)
```

这样自然语言报警指令会映射到现有 `restore_active_alarm_report()`、`write_cancelled_alarm_report()` 和 `write_alarm_control_command()`，无需移动或修改这些函数。

## 5. 已完成的集成收口

本次已经完成：

1. `web_app.py` 使用进程级 `AgentService`，避免每次请求重复初始化。
2. 已提供 `/api/agent/chat` 与 `/api/agent/history`。
3. 最终页面已 include 聊天组件及其 CSS/JS。
4. 已用最终视频检测结果核对 `num_events`、`class_counts`、`overall_risk` 和报告路径字段。
5. 已验证实际模型检测、上一轮查询和 SQLite 历史恢复链路。

集成没有再次修改 YOLO 训练或视频检测内部算法。
