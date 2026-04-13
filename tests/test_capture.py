from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from image_classifier.capture import DEFAULT_CAPTURES_DIR, save_frame


class TestSaveFrame:
    """Tests for the save_frame helper."""

    @staticmethod
    def _make_frame(width: int = 64, height: int = 48) -> NDArray[np.uint8]:
        return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    def test_creates_output_dir_and_saves_png(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "out"
        frame = self._make_frame()

        result = save_frame(frame, output_dir=output_dir)

        assert result is not None
        assert result.exists()
        assert result.suffix == ".png"
        assert result.parent == output_dir

    def test_saved_image_is_readable(self, tmp_path: Path) -> None:
        frame = self._make_frame()

        result = save_frame(frame, output_dir=tmp_path)

        assert result is not None
        loaded = cv2.imread(str(result))
        assert loaded is not None
        assert loaded.shape == frame.shape

    def test_filename_contains_timestamp(self, tmp_path: Path) -> None:
        frame = self._make_frame()

        result = save_frame(frame, output_dir=tmp_path)

        assert result is not None
        assert result.name.startswith("capture_")
        # 8 digits date + _ + 6 digits time = "YYYYMMDD_HHMMSS"
        stem = result.stem.removeprefix("capture_")
        assert len(stem) == 15  # "YYYYMMDD_HHMMSS"

    def test_returns_none_on_write_failure(self, tmp_path: Path) -> None:
        frame = self._make_frame()

        with patch("image_classifier.capture.cv2.imwrite", return_value=False):
            result = save_frame(frame, output_dir=tmp_path)

        assert result is None

    def test_default_output_dir_is_captures(self) -> None:
        assert DEFAULT_CAPTURES_DIR == Path("captures")
