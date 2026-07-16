# 大模型 API 接入

大模型连接实现在 `agent/llm_api.py`。该文件支持提供 OpenAI-compatible `chat/completions` 接口的服务商，不在源码中保存真实密钥，也不改变 YOLO、风险规则或报警控制逻辑。

## 1. 配置密钥

在项目目录执行：

```powershell
Copy-Item .env.example .env
```

编辑 `.env`：

```dotenv
LLM_API_KEY=你的真实密钥
LLM_BASE_URL=服务商的OpenAI兼容v1地址
LLM_MODEL=服务商支持的模型名称
LLM_TIMEOUT_SECONDS=60
LLM_MAX_TOKENS=1600
LLM_JSON_MODE=true
LLM_PLANNER_MODE=hybrid
```

`.env` 已被仓库根目录的 `.gitignore` 忽略。不要把真实密钥写进 `agent/llm_api.py`、测试、聊天消息或前端 JavaScript。

如果服务商不支持 `response_format={"type":"json_object"}`，设置 `LLM_JSON_MODE=false`。模型仍会被提示只返回 JSON，但兼容性取决于服务商模型。

`LLM_PLANNER_MODE=hybrid` 为推荐模式：简单明确命令走本地规则，复杂或未知任务调用模型。设置为 `always` 后，除帮助命令外的所有任务都会先调用模型规划。

## 2. 运行

直接运行大模型版智能体：

```powershell
python -m agent.llm_api "查询一号线今天上午的高风险记录并生成汇总"
```

带图片：

```powershell
python -m agent.llm_api "检测这张图片并记录到一号线" `
  --image data/demo/belt.jpg `
  --line-id line-1
```

带视频：

```powershell
python -m agent.llm_api "按每秒4帧检测这段视频并生成风险结果" `
  --video data/demo/belt.mp4 `
  --video-start-time 2026-07-16T08:00:00+08:00 `
  --line-id line-1
```

简单且明确的原有命令仍由本地规则直接处理，不产生 API 请求。规则无法表达的任务，以及包含时间段、线路、ROI、阈值、抽帧、复核或多步骤编排的任务，才交给大模型规划。

## 3. Python 接入

```python
from agent.llm_api import create_llm_enabled_service

service = create_llm_enabled_service()
result = service.chat(
    "检测今天上午8点到9点的一号线视频并汇总风险",
    session_id="operator-1",
    context={
        "video_path": "data/line-1-0800.mp4",
        "video_start_time": "2026-07-16T08:00:00+08:00",
        "line_id": "line-1",
    },
)
```

大模型输出计划结构：

```json
{
  "summary": "检测指定视频",
  "needs_clarification": false,
  "clarification": "",
  "steps": [
    {
      "skill_name": "detect-video",
      "arguments": {
        "parameters": {"sample_fps": 4.0}
      }
    }
  ]
}
```

后续步骤可通过 `$steps.0.data.detection_id` 引用前一步结果。执行层最多接受 6 步，并再次校验 Skill 名称和参数。

## 4. 安全边界

- 大模型不能执行任意 Python 函数，只能选择注册表中的 Skill。
- 图片和视频检测仍调用现有确定性检测函数。
- 风险等级和处置建议仍由 `task3_alarm` 规则引擎生成。
- 报警确认、取消和检测复核必须在原始用户文本中出现明确动作，模型计划不能绕过。
- 缺少录像实际路径时，大模型必须请求补充信息；历史元数据不会被误当成录像文件库。
- 请求仅发送文本、Skill catalog、最近对话文本和本轮路径/参数，不发送图片或视频二进制。

## 5. 后续 Web 接入

等接口稳定后，在 `web_app.py` 创建进程级 `create_llm_enabled_service()` 即可沿用现有 `/api/agent/chat`。密钥只保存在服务器 `.env`，绝不能下发到浏览器。
