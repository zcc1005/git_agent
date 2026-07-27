from __future__ import annotations

import os
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RUNS_DIR = PROJECT_ROOT / "runs"
_PORTABLE_PROJECT_DIRS = (
    "config",
    "data",
    "models",
    "outputs",
    "runs",
    "static",
    "templates",
)
_LEGACY_RELOCATABLE_DIRS = ("outputs",)


def portable_project_path(path_value: str | Path) -> str:
    """Return a stable project-relative path when a file belongs to the project.

    Persisted paths must not contain a workstation or container root.  The
    fallback for external files remains the original value because callers may
    intentionally use a mounted path outside the project.
    """

    raw_value = str(path_value).strip()
    if not raw_value:
        return ""

    normalized = raw_value.replace("\\", "/")
    candidate = Path(raw_value).expanduser()
    try:
        return candidate.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        pass

    # A Windows path is not considered absolute when read on Linux.  This also
    # migrates legacy database values such as C:/old/project/outputs/file.jpg.
    lowered = normalized.lower()
    for directory in _PORTABLE_PROJECT_DIRS:
        if lowered == directory or lowered.startswith(f"{directory}/"):
            return normalized
    for directory in _LEGACY_RELOCATABLE_DIRS:
        marker = f"/{directory}/"
        marker_index = lowered.rfind(marker)
        if marker_index >= 0:
            return normalized[marker_index + 1 :]
    return raw_value


def resolve_runtime_path(
    path_value: str | Path,
    *,
    require_project_path: bool = False,
) -> Path:
    """Resolve a stored relative or legacy absolute path for this deployment."""

    raw_value = str(path_value).strip()
    if not raw_value:
        raise ValueError("path_value cannot be empty")

    portable_value = portable_project_path(raw_value)
    is_windows_absolute = bool(re.match(r"^[A-Za-z]:[\\/]", raw_value))
    if portable_value != raw_value:
        resolved = (PROJECT_ROOT / Path(portable_value)).resolve()
        if require_project_path:
            try:
                resolved.relative_to(PROJECT_ROOT.resolve())
            except ValueError as exc:
                raise ValueError(f"path is outside the project: {path_value}") from exc
        return resolved

    original = Path(raw_value).expanduser()
    if original.is_absolute() and not require_project_path:
        try:
            original.resolve().relative_to(PROJECT_ROOT.resolve())
        except (OSError, ValueError):
            return original

    portable = Path(portable_value)
    if portable.is_absolute() and not is_windows_absolute:
        if require_project_path:
            raise ValueError(f"path is outside the project: {path_value}")
        return portable

    resolved = (PROJECT_ROOT / portable).resolve()
    if require_project_path:
        try:
            resolved.relative_to(PROJECT_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"path is outside the project: {path_value}") from exc
    return resolved


def resolve_project_path(path_value: str | Path | None, default: str | Path | None = None) -> Path:
    raw_value = path_value if path_value not in (None, "") else default
    if raw_value is None:
        raise ValueError("path_value and default cannot both be empty")

    path = Path(os.path.expandvars(str(raw_value))).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def env_path(env_name: str, default: str | Path) -> Path:
    return resolve_project_path(os.getenv(env_name), default)


def env_text(env_name: str, default: str) -> str:
    return os.getenv(env_name, default)


YOLO_DATA_DIR = env_path("YOLO_DATA_DIR", DATA_DIR / "yolo_yiwu")
YOLO_RUN_DIR = env_path("YOLO_RUN_DIR", RUNS_DIR / "yolo")
YOLO_RUN_NAME = env_text("YOLO_RUN_NAME", "yiwu_yolov8s_4class")
YOLO_MODEL_PATH = env_path(
    "YOLO_MODEL_PATH",
    YOLO_RUN_DIR / YOLO_RUN_NAME / "weights" / "best.pt",
)
YOLO_DEVICE = env_text("YOLO_DEVICE", "").strip()
VIDEO_SOURCES_PATH = env_path(
    "VIDEO_SOURCES_PATH",
    CONFIG_DIR / "video_sources.json",
)
