from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.typing import NDArray

from image_classifier.camera import capture_from_webcam

_CAMERA_MODULE = "image_classifier.camera"


class TestCaptureFromWebcam:
    """Tests for the webcam capture loop (camera and UI mocked)."""

    @staticmethod
    def _make_frame() -> NDArray[np.uint8]:
        return np.zeros((48, 64, 3), dtype=np.uint8)

    def test_exits_when_camera_cannot_open(self) -> None:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        with (
            patch(f"{_CAMERA_MODULE}.cv2.VideoCapture", return_value=mock_cap),
            pytest.raises(SystemExit, match="1"),
        ):
            capture_from_webcam()

    def test_quit_key_ends_loop(self) -> None:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, self._make_frame())

        with (
            patch(f"{_CAMERA_MODULE}.cv2.VideoCapture", return_value=mock_cap),
            patch(f"{_CAMERA_MODULE}.cv2.namedWindow"),
            patch(f"{_CAMERA_MODULE}.cv2.imshow"),
            patch(f"{_CAMERA_MODULE}.cv2.waitKey", return_value=ord("q")),
            patch(f"{_CAMERA_MODULE}.cv2.destroyAllWindows"),
        ):
            capture_from_webcam()

        mock_cap.release.assert_called_once()

    def test_save_key_triggers_classify_and_save(self, tmp_path: Path) -> None:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        frame = self._make_frame()
        mock_cap.read.return_value = (True, frame)

        # First call: 's' to save, second call: 'q' to quit.
        wait_returns = iter([ord("s"), ord("q")])
        annotated = self._make_frame()

        with (
            patch(f"{_CAMERA_MODULE}.cv2.VideoCapture", return_value=mock_cap),
            patch(f"{_CAMERA_MODULE}.cv2.namedWindow"),
            patch(f"{_CAMERA_MODULE}.cv2.imshow"),
            patch(f"{_CAMERA_MODULE}.cv2.waitKey", side_effect=wait_returns),
            patch(f"{_CAMERA_MODULE}.cv2.destroyAllWindows"),
            patch(f"{_CAMERA_MODULE}.classify_food", return_value=("Pizza", 0.95)) as mock_classify,
            patch(f"{_CAMERA_MODULE}.annotate_image", return_value=annotated) as mock_annotate,
            patch(f"{_CAMERA_MODULE}.save_frame", return_value=tmp_path / "test.png") as mock_save,
        ):
            capture_from_webcam()

        mock_classify.assert_called_once()
        mock_annotate.assert_called_once_with(frame, "Pizza", 0.95)
        mock_save.assert_called_once_with(annotated)

    def test_read_failure_breaks_loop(self) -> None:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)

        with (
            patch(f"{_CAMERA_MODULE}.cv2.VideoCapture", return_value=mock_cap),
            patch(f"{_CAMERA_MODULE}.cv2.namedWindow"),
            patch(f"{_CAMERA_MODULE}.cv2.imshow"),
            patch(f"{_CAMERA_MODULE}.cv2.destroyAllWindows"),
        ):
            capture_from_webcam()

        mock_cap.release.assert_called_once()

    def test_save_failure_prints_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, self._make_frame())

        # First call: 's' to save (will fail), second call: 'q' to quit.
        wait_returns = iter([ord("s"), ord("q")])

        with (
            patch(f"{_CAMERA_MODULE}.cv2.VideoCapture", return_value=mock_cap),
            patch(f"{_CAMERA_MODULE}.cv2.namedWindow"),
            patch(f"{_CAMERA_MODULE}.cv2.imshow"),
            patch(f"{_CAMERA_MODULE}.cv2.waitKey", side_effect=wait_returns),
            patch(f"{_CAMERA_MODULE}.cv2.destroyAllWindows"),
            patch(f"{_CAMERA_MODULE}.classify_food", return_value=("Pizza", 0.95)),
            patch(f"{_CAMERA_MODULE}.annotate_image", return_value=self._make_frame()),
            patch(f"{_CAMERA_MODULE}.save_frame", return_value=None),
        ):
            capture_from_webcam()

        assert "Error: could not write capture" in capsys.readouterr().err
