from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4


def save_content_addressed_upload(file, upload_dir: Path, suffix: str) -> Path:
    """Save an upload once and reuse it when identical content already exists."""

    upload_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = upload_dir / f".upload-{uuid4().hex}.tmp"
    try:
        file.save(temporary_path)
        digest = hashlib.sha256()
        with temporary_path.open("rb") as uploaded:
            for chunk in iter(lambda: uploaded.read(1024 * 1024), b""):
                digest.update(chunk)
        saved_path = upload_dir / f"{digest.hexdigest()[:24]}{suffix.lower()}"
        if saved_path.is_file():
            temporary_path.unlink()
        else:
            temporary_path.replace(saved_path)
        return saved_path
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
