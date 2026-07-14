from __future__ import annotations

import argparse
import base64
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from werkzeug.datastructures import FileStorage


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

UPLOAD_DIR = PROJECT_ROOT / "outputs" / "uploaded_videos"
EXTRACTED_FRAMES_DIR = PROJECT_ROOT / "outputs" / "extracted_frames"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".mpeg", ".mpg"}


def _safe_filename(name: str) -> str:
    """Return a filesystem-safe, non-empty filename stem."""
    cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in name)
    return cleaned.strip("_") or "video"


def _ffmpeg_executable() -> str | None:
    """Find ffmpeg without depending on a path from another developer's machine."""
    executable = shutil.which("ffmpeg")
    if executable:
        return executable

    local_candidates = (
        PROJECT_ROOT.parent / "ffmpeg" / "bin" / "ffmpeg.exe",
        PROJECT_ROOT / "ffmpeg" / "bin" / "ffmpeg.exe",
    )
    for candidate in local_candidates:
        if candidate.is_file():
            return str(candidate)

    return None


def _extract_with_opencv(
    video_path: Path,
    output_img_dir: Path,
    prefix: str,
    frame_fps: float,
    img_suffix: str,
) -> list[Path]:
    """Extract frames without ffmpeg when OpenCV is available."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Neither ffmpeg nor OpenCV is available for frame extraction. "
            "Install ffmpeg or opencv-python."
        ) from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 25.0
    frame_interval = source_fps / frame_fps
    next_frame_at = 0.0
    source_index = 0
    output_index = 1
    generated: list[Path] = []

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if source_index + 1e-9 >= next_frame_at:
                output_path = output_img_dir / f"{prefix}_frame_{output_index:06d}.{img_suffix}"
                if not cv2.imwrite(str(output_path), frame):
                    raise RuntimeError(f"OpenCV could not write frame: {output_path}")
                generated.append(output_path)
                output_index += 1
                next_frame_at += frame_interval
            source_index += 1
    finally:
        capture.release()

    return generated


def review_extracted_images(image_paths: list[Path]) -> None:
    """Open a small local review window for keeping or deleting extracted images."""
    images = [path for path in image_paths if path.exists()]
    if not images:
        print("No extracted images are available for review.")
        return

    try:
        import tkinter as tk
        from tkinter import messagebox
    except ImportError as exc:
        print(f"Image review is unavailable: {exc}")
        return

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"Image review window could not be opened: {exc}")
        return

    root.title("Extracted frame review")
    root.geometry("1000x760")
    root.minsize(700, 520)
    state: dict[str, object] = {"index": 0, "photo": None}

    image_label = tk.Label(root, bg="#202124")
    image_label.pack(fill="both", expand=True, padx=12, pady=(12, 6))
    info = tk.StringVar()
    tk.Label(root, textvariable=info).pack(fill="x", padx=12, pady=6)
    buttons = tk.Frame(root)
    buttons.pack(pady=(0, 12))

    def show_current() -> None:
        if not images:
            messagebox.showinfo("Review complete", "All images have been deleted.", parent=root)
            root.destroy()
            return

        index = min(int(state["index"]), len(images) - 1)
        state["index"] = index
        current = images[index]
        ffmpeg = _ffmpeg_executable()
        if not ffmpeg:
            messagebox.showerror(
                "Preview unavailable",
                "Install ffmpeg and add its bin directory to PATH to review frames in this window.",
                parent=root,
            )
            return
        try:
            converted = subprocess.run(
                [
                    ffmpeg,
                    "-v",
                    "error",
                    "-i",
                    str(current),
                    "-vf",
                    "scale=940:620:force_original_aspect_ratio=decrease",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "png",
                    "-",
                ],
                check=True,
                capture_output=True,
            )
            png_data = base64.b64encode(converted.stdout).decode("ascii")
            state["photo"] = tk.PhotoImage(data=png_data, format="png")
        except (subprocess.CalledProcessError, FileNotFoundError, tk.TclError) as exc:
            messagebox.showerror("Preview failed", f"Could not show {current.name}\n{exc}", parent=root)
            return

        image_label.configure(image=state["photo"])
        info.set(f"{index + 1} / {len(images)}    {current.name}")

    def move(offset: int) -> None:
        if images:
            state["index"] = (int(state["index"]) + offset) % len(images)
            show_current()

    def delete_current() -> None:
        current = images[int(state["index"])]
        if not messagebox.askyesno("Delete frame", f"Delete {current.name}?", parent=root):
            return
        try:
            current.unlink()
        except OSError as exc:
            messagebox.showerror("Delete failed", str(exc), parent=root)
            return
        images.pop(int(state["index"]))
        show_current()

    tk.Button(buttons, text="Previous", width=12, command=lambda: move(-1)).pack(side="left", padx=5)
    tk.Button(buttons, text="Next", width=12, command=lambda: move(1)).pack(side="left", padx=5)
    tk.Button(buttons, text="Delete frame", width=16, command=delete_current).pack(side="left", padx=5)
    tk.Button(buttons, text="Finish", width=12, command=root.destroy).pack(side="left", padx=5)
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
    """Extract evenly sampled frames from one video and return their paths."""
    video_path = video_path.resolve()
    if not video_path.is_file():
        print(f"Video file does not exist: {video_path}")
        return []
    if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        print(f"Unsupported video format: {video_path.suffix}")
        return []
    if frame_fps <= 0:
        raise ValueError("--fps must be greater than 0.")

    output_img_dir.mkdir(parents=True, exist_ok=True)
    prefix = _safe_filename(video_path.stem)
    output_pattern = str(output_img_dir / f"{prefix}_frame_%06d.{img_suffix}")
    for old_frame in output_img_dir.glob(f"{prefix}_frame_*.{img_suffix}"):
        old_frame.unlink()

    ffmpeg = _ffmpeg_executable()
    try:
        if ffmpeg:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(video_path),
                    "-vf",
                    f"fps={frame_fps}",
                    output_pattern,
                ],
                check=True,
            )
            generated = sorted(output_img_dir.glob(f"{prefix}_frame_*.{img_suffix}"))
        else:
            print("ffmpeg was not found; using OpenCV for frame extraction.")
            generated = _extract_with_opencv(
                video_path, output_img_dir, prefix, frame_fps, img_suffix
            )
    except (subprocess.CalledProcessError, RuntimeError) as exc:
        print(f"Frame extraction failed: {exc}")
        return []
    print(
        f"Frame extraction complete: {video_path.name}; "
        f"{len(generated)} frames saved to {output_img_dir}"
    )
    return generated


def batch_extract_folder_videos(
    video_folder: Path,
    output_dir: Path = EXTRACTED_FRAMES_DIR,
    frame_fps: float = 0.5,
) -> list[Path]:
    """Recursively extract frames from all supported videos in a folder."""
    if not video_folder.is_dir():
        print(f"Video directory does not exist: {video_folder}")
        return []

    videos = sorted(
        path
        for path in video_folder.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        print(f"No supported videos found in: {video_folder}")
        return []

    target_dir = output_dir / _safe_filename(video_folder.name)
    print(f"Found {len(videos)} video(s). Extracting frames to {target_dir}")
    generated: list[Path] = []
    for video in videos:
        print(f"Processing: {video}")
        generated.extend(extract_frames_from_video(video, target_dir, frame_fps))
    return generated


def save_uploaded_video(file: FileStorage, upload_dir: Path = UPLOAD_DIR) -> Path:
    """Store a browser-uploaded video safely and return its project-local path."""
    original_name = Path(file.filename or "").name
    suffix = Path(original_name).suffix.lower()
    if not original_name:
        raise ValueError("Choose a video file before uploading.")
    if suffix not in VIDEO_EXTENSIONS:
        allowed = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise ValueError(f"Unsupported video format. Allowed formats: {allowed}")

    upload_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    saved_path = upload_dir / f"{timestamp}_{_safe_filename(Path(original_name).stem)}{suffix}"
    file.save(saved_path)
    return saved_path


def create_upload_app():
    """Create a small Flask app used only for video upload and frame extraction."""
    try:
        from flask import Flask, jsonify, render_template_string, request
    except ImportError as exc:
        raise RuntimeError("Upload mode requires Flask. Install dependencies first.") from exc

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024

    page = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Video frame extraction</title><style>
body{margin:0;background:#f4f7f7;color:#1f2933;font-family:"Microsoft YaHei",Arial,sans-serif}
main{width:min(720px,calc(100% - 32px));margin:48px auto}h1{margin:0 0 8px;font-size:26px}p{color:#52606d;line-height:1.6}
form,#result{margin-top:24px;padding:24px;border:1px solid #d9e2e7;border-radius:8px;background:#fff}
label{display:block;margin:16px 0 8px;font-weight:700}input,select,button{font:inherit}input,select{width:100%;box-sizing:border-box;padding:10px;border:1px solid #aab7c1;border-radius:6px}
button{margin-top:22px;padding:10px 18px;border:0;border-radius:6px;color:#fff;background:#0f766e;cursor:pointer}button:disabled{opacity:.65;cursor:wait}
#result{display:none;white-space:pre-wrap;line-height:1.65}.error{color:#b42318}
</style></head><body><main><h1>Upload a video and extract frames</h1><p>The uploaded file is stored in this project's outputs/uploaded_videos directory. Extracted images are written to outputs/extracted_frames.</p>
<form id="uploadForm"><label for="video">Video file</label><input id="video" name="video" type="file" accept="video/*,.avi,.mkv,.flv,.wmv,.mpeg,.mpg" required>
<label for="fps">Frames per second</label><input id="fps" name="fps" type="number" min="0.01" step="0.01" value="0.5" required><button id="submit" type="submit">Upload and extract</button></form>
<div id="result"></div></main><script>
const form=document.getElementById('uploadForm'),result=document.getElementById('result'),submit=document.getElementById('submit');
form.addEventListener('submit',async(event)=>{event.preventDefault();submit.disabled=true;submit.textContent='Uploading and extracting...';result.style.display='block';result.className='';result.textContent='Working...';try{const response=await fetch('/api/extract',{method:'POST',body:new FormData(form)});const data=await response.json();if(!response.ok||!data.ok)throw new Error(data.error||'Extraction failed');result.textContent=`Done. ${data.frame_count} frame(s) extracted.\\nUploaded video: ${data.video_path}\\nFrame directory: ${data.output_dir}`;}catch(error){result.className='error';result.textContent=error.message;}finally{submit.disabled=false;submit.textContent='Upload and extract';}});
</script></body></html>"""

    @app.get("/")
    def index():
        return render_template_string(page)

    @app.post("/api/extract")
    def extract_uploaded_video():
        try:
            uploaded_file = request.files.get("video")
            if uploaded_file is None:
                raise ValueError("Choose a video file before uploading.")
            fps = float(request.form.get("fps", "0.5"))

            video_path = save_uploaded_video(uploaded_file)
            output_dir = EXTRACTED_FRAMES_DIR / video_path.stem
            frames = extract_frames_from_video(video_path, output_dir, fps)
            if not frames:
                raise RuntimeError("No frames were created. Check the ffmpeg output above.")
            return jsonify(
                {
                    "ok": True,
                    "frame_count": len(frames),
                    "video_path": str(video_path.relative_to(PROJECT_ROOT)),
                    "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
                }
            )
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Video frame extraction utility")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", type=Path, help="Path to one local video file")
    source.add_argument("--video-dir", type=Path, help="Directory containing local video files")
    source.add_argument(
        "--serve",
        action="store_true",
        help="Start the browser upload page (the default when no video source is given)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXTRACTED_FRAMES_DIR,
        help="Directory for extracted frame images",
    )
    parser.add_argument("--fps", type=float, default=0.5, help="Frames per second to extract")
    parser.add_argument("--no-review", action="store_true", help="Do not open the local frame review window")
    parser.add_argument("--host", default="127.0.0.1", help="Upload server host")
    parser.add_argument("--port", type=int, default=5050, help="Upload server port")
    args = parser.parse_args()

    if args.serve or (args.video is None and args.video_dir is None):
        print(f"Open http://{args.host}:{args.port} in a browser to upload a video.")
        create_upload_app().run(host=args.host, port=args.port, debug=False)
        return

    if args.video:
        output_dir = args.output_dir / _safe_filename(args.video.stem)
        generated = extract_frames_from_video(args.video, output_dir, args.fps)
    else:
        generated = batch_extract_folder_videos(args.video_dir, args.output_dir, args.fps)
    if generated and not args.no_review:
        review_extracted_images(generated)


if __name__ == "__main__":
    main()
