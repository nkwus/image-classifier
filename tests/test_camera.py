from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.typing import NDArray

from image_classifier.camera import capture_from_webcam, _classify_worker

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
            patch(f"{_CAMERA_MODULE}.threading.Thread") as mock_thread_cls,
        ):
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread
            capture_from_webcam()

        mock_cap.release.assert_called_once()
        mock_thread.join.assert_called_once()

    def test_save_key_enqueues_frame(self) -> None:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        frame = self._make_frame()
        mock_cap.read.return_value = (True, frame)

        wait_returns = iter([ord("s"), ord("q")])

        with (
            patch(f"{_CAMERA_MODULE}.cv2.VideoCapture", return_value=mock_cap),
            patch(f"{_CAMERA_MODULE}.cv2.namedWindow"),
            patch(f"{_CAMERA_MODULE}.cv2.imshow"),
            patch(f"{_CAMERA_MODULE}.cv2.waitKey", side_effect=wait_returns),
            patch(f"{_CAMERA_MODULE}.cv2.destroyAllWindows"),
            patch(f"{_CAMERA_MODULE}.threading.Thread") as mock_thread_cls,
            patch(f"{_CAMERA_MODULE}.queue.Queue") as mock_queue_cls,
        ):
            mock_queue = MagicMock()
            mock_queue_cls.return_value = mock_queue
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread
            capture_from_webcam()

        # First put: the copied frame; second put: the sentinel (None).
        assert mock_queue.put.call_count == 2
        # The first call should be a frame (numpy array copy).
        first_arg = mock_queue.put.call_args_list[0][0][0]
        np.testing.assert_array_equal(first_arg, frame)
        # The second call should be the sentinel.
        assert mock_queue.put.call_args_list[1][0][0] is None

    def test_read_failure_breaks_loop(self) -> None:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)

        with (
            patch(f"{_CAMERA_MODULE}.cv2.VideoCapture", return_value=mock_cap),
            patch(f"{_CAMERA_MODULE}.cv2.namedWindow"),
            patch(f"{_CAMERA_MODULE}.cv2.imshow"),
            patch(f"{_CAMERA_MODULE}.cv2.destroyAllWindows"),
            patch(f"{_CAMERA_MODULE}.threading.Thread") as mock_thread_cls,
        ):
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread
            capture_from_webcam()

        mock_cap.release.assert_called_once()


class TestClassifyWorker:
    """Tests for the background classification thread."""

    @staticmethod
    def _make_frame() -> NDArray[np.uint8]:
        return np.zeros((48, 64, 3), dtype=np.uint8)

    def test_processes_frame_and_saves(self, tmp_path: Path) -> None:
        import queue as _queue

        q: _queue.Queue[NDArray[np.uint8] | None] = _queue.Queue()
        frame = self._make_frame()
        q.put(frame)
        q.put(None)  # sentinel to stop

        annotated = self._make_frame()
        with (
            patch(f"{_CAMERA_MODULE}.classify_food", return_value=("Pizza", 0.95)) as mock_classify,
            patch(f"{_CAMERA_MODULE}.annotate_image", return_value=annotated) as mock_annotate,
            patch(f"{_CAMERA_MODULE}.save_frame", return_value=tmp_path / "test.png") as mock_save,
        ):
            _classify_worker(q)

        mock_classify.assert_called_once()
        mock_annotate.assert_called_once_with(frame, "Pizza", 0.95)
        mock_save.assert_called_once_with(annotated)

    def test_save_failure_prints_error(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        import queue as _queue

        q: _queue.Queue[NDArray[np.uint8] | None] = _queue.Queue()
        q.put(self._make_frame())
        q.put(None)

        with (
            patch(f"{_CAMERA_MODULE}.classify_food", return_value=("Pizza", 0.95)),
            patch(f"{_CAMERA_MODULE}.annotate_image", return_value=self._make_frame()),
            patch(f"{_CAMERA_MODULE}.save_frame", return_value=None),
        ):
            _classify_worker(q)

        assert "Error: could not write capture" in capsys.readouterr().err

    def test_classification_exception_prints_error(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        import queue as _queue

        q: _queue.Queue[NDArray[np.uint8] | None] = _queue.Queue()
        q.put(self._make_frame())
        q.put(None)

        with patch(
            f"{_CAMERA_MODULE}.classify_food",
            side_effect=RuntimeError("model exploded"),
        ):
            _classify_worker(q)

        assert "Error during classification: model exploded" in capsys.readouterr().err

    def test_sentinel_stops_worker(self) -> None:
        import queue as _queue

        q: _queue.Queue[NDArray[np.uint8] | None] = _queue.Queue()
        q.put(None)

        # Should return immediately without calling classify_food.
        with patch(f"{_CAMERA_MODULE}.classify_food") as mock_classify:
            _classify_worker(q)

        mock_classify.assert_not_called()
