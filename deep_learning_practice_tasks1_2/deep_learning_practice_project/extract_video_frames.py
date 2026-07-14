from __future__ import annotations

import argparse
import base64
import shutil
import subprocess
import sys
from pathlib import Path


# 该脚本位于项目根目录。正确加入项目根路径，避免找不到 task2_yolo。
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task2_yolo.yolo_config import YOLO_DATA_DIR


DEFAULT_VIDEO_DIR = Path(r"C:\Users\荣\Desktop\videos")
LOCAL_FFMPEG = Path(
    r"C:\Users\荣\Desktop\ffmpeg-2026-07-09-git-8de8405796-full_build"
    r"\ffmpeg-2026-07-09-git-8de8405796-full_build\bin\ffmpeg.exe"
)
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".mpeg", ".mpg"}


def _safe_filename(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return cleaned.strip("_") or "video"


def _ffmpeg_executable() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    if LOCAL_FFMPEG.is_file():
        return str(LOCAL_FFMPEG)
    raise FileNotFoundError(
        "未找到 ffmpeg。请安装 ffmpeg，并将 ffmpeg.exe 所在的 bin 目录加入 PATH。"
    )


def review_extracted_images(image_paths: list[Path]) -> None:
    """弹窗逐张验证抽帧照片，可删除不合格照片。"""
    images = [path for path in image_paths if path.exists()]
    if not images:
        print("本次没有生成可验证的照片。")
        return

    try:
        import tkinter as tk
        from tkinter import messagebox
    except ImportError as exc:
        print(f"无法打开照片验证弹窗：{exc}")
        return
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"无法打开照片验证弹窗：{exc}")
        return

    root.title("抽帧照片验证")
    root.geometry("1000x760")
    root.minsize(700, 520)
    state = {"index": 0, "photo": None}

    image_label = tk.Label(root, bg="#202124")
    image_label.pack(fill="both", expand=True, padx=12, pady=(12, 6))
    info = tk.StringVar()
    tk.Label(root, textvariable=info).pack(fill="x", padx=12, pady=6)
    buttons = tk.Frame(root)
    buttons.pack(pady=(0, 12))

    def show_current() -> None:
        if not images:
            messagebox.showinfo("验证完成", "所有照片已删除。", parent=root)
            root.destroy()
            return
        state["index"] = min(state["index"], len(images) - 1)
        current = images[state["index"]]
        try:
            converted = subprocess.run(
                [
                    _ffmpeg_executable(), "-v", "error", "-i", str(current),
                    "-vf", "scale=940:620:force_original_aspect_ratio=decrease",
                    "-f", "image2pipe", "-vcodec", "png", "-",
                ],
                check=True,
                capture_output=True,
            )
            png_data = base64.b64encode(converted.stdout).decode("ascii")
            state["photo"] = tk.PhotoImage(data=png_data, format="png")
        except (subprocess.CalledProcessError, FileNotFoundError, tk.TclError) as exc:
            messagebox.showerror("预览失败", f"无法显示 {current.name}\n{exc}", parent=root)
            return
        image_label.configure(image=state["photo"])
        info.set(f"{state['index'] + 1} / {len(images)}    {current.name}")

    def move(offset: int) -> None:
        if images:
            state["index"] = (state["index"] + offset) % len(images)
            show_current()

    def delete_current() -> None:
        current = images[state["index"]]
        if not messagebox.askyesno("删除照片", f"确定删除这张照片吗？\n{current.name}", parent=root):
            return
        try:
            current.unlink()
        except OSError as exc:
            messagebox.showerror("删除失败", str(exc), parent=root)
            return
        images.pop(state["index"])
        show_current()

    tk.Button(buttons, text="上一张", width=12, command=lambda: move(-1)).pack(side="left", padx=5)
    tk.Button(buttons, text="下一张", width=12, command=lambda: move(1)).pack(side="left", padx=5)
    tk.Button(buttons, text="删除不合格照片", width=16, command=delete_current).pack(side="left", padx=5)
    tk.Button(buttons, text="完成验证", width=12, command=root.destroy).pack(side="left", padx=5)
    root.bind("<Left>", lambda _event: move(-1))
    root.bind("<Right>", lambda _event: move(1))
    show_current()
    root.mainloop()


def extract_frames_from_video(
    video_path: Path,
    output_img_dir: Path,
    frame_fps: float = 0.5,
    img_suffix: str = "jpg",
) -> list[Path]:
    """按程序指定的频率均匀抽帧，返回本次的照片路径。"""
    video_path = video_path.resolve()
    if not video_path.is_file():
        print(f"错误：视频文件不存在 -> {video_path}")
        return []
    if frame_fps <= 0:
        raise ValueError("抽帧频率 --fps 必须大于 0。")

    output_img_dir.mkdir(parents=True, exist_ok=True)
    prefix = _safe_filename(video_path.stem)
    output_pattern = str(output_img_dir / f"{prefix}_frame_%06d.{img_suffix}")
    # 清理该视频上次的抽帧，避免更改 fps 后混入旧照片。
    for old_frame in output_img_dir.glob(f"{prefix}_frame_*.{img_suffix}"):
        old_frame.unlink()
    try:
        subprocess.run(
            [
                _ffmpeg_executable(),
                "-y",
                "-i", str(video_path),
                "-vf", f"fps={frame_fps}",
                output_pattern,
            ],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"FFmpeg 抽帧失败：{exc}")
        return []

    generated = sorted(output_img_dir.glob(f"{prefix}_frame_*.{img_suffix}"))
    print(f"抽帧完成：{video_path.name}，共 {len(generated)} 张，输出目录：{output_img_dir}")
    return generated


def batch_extract_folder_videos(video_folder: Path, split_name: str, frame_fps: float = 0.5) -> list[Path]:
    """递归处理视频目录及子目录中的所有视频。"""
    if not video_folder.is_dir():
        print(f"错误：视频目录不存在 -> {video_folder}")
        return []
    videos = sorted(
        path for path in video_folder.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        print(f"文件夹 {video_folder} 及其子目录中未找到视频文件。")
        return []

    target_dir = YOLO_DATA_DIR / "images" / split_name
    print(f"共检测到 {len(videos)} 个视频，开始抽帧至 {target_dir}")
    generated: list[Path] = []
    for video in videos:
        print(f"\n===== 正在处理：{video} =====")
        generated.extend(extract_frames_from_video(video, target_dir, frame_fps))
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="视频抽帧与照片验证工具")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", type=Path, help="单个视频文件路径")
    source.add_argument(
        "--video_dir", type=Path, default=DEFAULT_VIDEO_DIR,
        help=f"批量视频目录（默认：{DEFAULT_VIDEO_DIR}）",
    )
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--fps", type=float, default=0.5, help="抽帧频率，0.5 表示每 2 秒 1 张")
    parser.add_argument("--no_review", action="store_true", help="抽帧后不打开照片验证弹窗")
    args = parser.parse_args()

    if args.video:
        generated = extract_frames_from_video(args.video, YOLO_DATA_DIR / "images" / args.split, args.fps)
    else:
        generated = batch_extract_folder_videos(args.video_dir, args.split, args.fps)
    if generated and not args.no_review:
        review_extracted_images(generated)


if __name__ == "__main__":
    main()
