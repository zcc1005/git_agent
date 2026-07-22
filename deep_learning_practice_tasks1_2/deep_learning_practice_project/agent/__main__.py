from __future__ import annotations

import argparse
import json
from pathlib import Path

from .service import create_default_service


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="工业异物检测智能体命令行入口")
    parser.add_argument("message", help="自然语言指令")
    parser.add_argument("--session-id", default="cli", help="会话 ID")
    parser.add_argument("--video", type=Path, help="待检测视频")
    parser.add_argument("--video-start-time", help="视频开始时间（ISO 8601）")
    parser.add_argument("--db", type=Path, help="SQLite 历史数据库路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = {}
    if args.video:
        context["video_path"] = str(args.video)
    if args.video_start_time:
        context["video_start_time"] = args.video_start_time
    response = create_default_service(args.db).chat(
        args.message,
        session_id=args.session_id,
        context=context,
    )
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
