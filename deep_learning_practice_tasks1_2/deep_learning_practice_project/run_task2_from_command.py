"""便捷入口：读取 outputs/command.json，如果 command 为 go，则执行任务2检测。"""

# 允许直接用 PyCharm/命令行运行本文件，而不必使用 python -m。
import sys
from pathlib import Path as _PathForSys
sys.path.append(str(_PathForSys(__file__).resolve().parent))

from task2_yolo.detect_yolo import main

if __name__ == "__main__":
    main()
