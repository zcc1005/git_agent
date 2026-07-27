from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from project_config import OUTPUTS_DIR, portable_project_path, resolve_runtime_path
from utils.media_files import save_content_addressed_upload


class MemoryUpload:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def save(self, path: Path) -> None:
        Path(path).write_bytes(self.content)


class MediaPathTests(unittest.TestCase):
    def test_project_file_is_persisted_as_relative_posix_path(self) -> None:
        path = OUTPUTS_DIR / "agent_inputs" / "images" / "example.jpg"

        self.assertEqual(
            portable_project_path(path),
            "outputs/agent_inputs/images/example.jpg",
        )

    def test_legacy_windows_output_path_maps_to_current_deployment(self) -> None:
        legacy = (
            r"C:\Users\old\project\outputs\agent_inputs\images\example.jpg"
        )

        self.assertEqual(
            portable_project_path(legacy),
            "outputs/agent_inputs/images/example.jpg",
        )
        self.assertEqual(
            resolve_runtime_path(legacy),
            OUTPUTS_DIR / "agent_inputs" / "images" / "example.jpg",
        )

    def test_identical_uploads_share_one_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            upload_dir = Path(temp_dir)
            first = save_content_addressed_upload(
                MemoryUpload(b"identical-content"),
                upload_dir,
                ".jpg",
            )
            second = save_content_addressed_upload(
                MemoryUpload(b"identical-content"),
                upload_dir,
                ".jpg",
            )

            self.assertEqual(first, second)
            self.assertEqual(
                [path for path in upload_dir.iterdir() if path.is_file()],
                [first],
            )


if __name__ == "__main__":
    unittest.main()
