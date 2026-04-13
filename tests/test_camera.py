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
            patch(f"{_CAMERA_MODULE}.ClassifierClient"),
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
            patch(f"{_CAMERA_MODULE}.ClassifierClient") as mock_client_cls,
            patch(f"{_CAMERA_MODULE}.threading.Thread") as mock_thread_cls,
        ):
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread
            capture_from_webcam()

        mock_cap.release.assert_called_once()
        mock_thread.join.assert_called_once()
        mock_client.close.assert_called_once()

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
            patch(f"{_CAMERA_MODULE}.ClassifierClient"),
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
            patch(f"{_CAMERA_MODULE}.ClassifierClient"),
            patch(f"{_CAMERA_MODULE}.threading.Thread") as mock_thread_cls,
        ):
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread
            capture_from_webcam()

        mock_cap.release.assert_called_once()


class TestClassifyWorker:
    """Tests for the background classification worker thread."""

    @staticmethod
    def _make_frame() -> NDArray[np.uint8]:
        return np.zeros((48, 64, 3), dtype=np.uint8)

    def test_processes_frame_and_saves(self, tmp_path: Path) -> None:
        import queue as _queue

        q: _queue.Queue[NDArray[np.uint8] | None] = _queue.Queue()
        frame = self._make_frame()
        q.put(frame)
        q.put(None)  # sentinel to stop

        mock_client = MagicMock()
        mock_submission = MagicMock()
        mock_submission.job_id = "job1"
        mock_client.submit.return_value = mock_submission

        # Create a valid PNG to return as annotated image
        import cv2
        ok, buf = cv2.imencode(".png", frame)
        annotated_bytes = buf.tobytes()

        mock_job = MagicMock()
        mock_job.label = "Pizza"
        mock_job.confidence = 0.95
        mock_client.wait_for_result.return_value = (annotated_bytes, mock_job)

        with patch(f"{_CAMERA_MODULE}.save_frame", return_value=tmp_path / "test.png"):
            _classify_worker(q, mock_client)

        mock_client.submit.assert_called_once()
        mock_client.wait_for_result.assert_called_once_with("job1", timeout=120.0)

    def test_save_failure_prints_error(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        import queue as _queue

        q: _queue.Queue[NDArray[np.uint8] | None] = _queue.Queue()
        frame = self._make_frame()
        q.put(frame)
        q.put(None)

        mock_client = MagicMock()
        mock_submission = MagicMock()
        mock_submission.job_id = "job1"
        mock_client.submit.return_value = mock_submission

        import cv2
        ok, buf = cv2.imencode(".png", frame)

        mock_job = MagicMock()
        mock_job.label = "Pizza"
        mock_job.confidence = 0.95
        mock_client.wait_for_result.return_value = (buf.tobytes(), mock_job)

        with patch(f"{_CAMERA_MODULE}.save_frame", return_value=None):
            _classify_worker(q, mock_client)

        assert "Error: could not write capture" in capsys.readouterr().err

    def test_wait_for_result_returns_none_prints_error(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        import queue as _queue

        q: _queue.Queue[NDArray[np.uint8] | None] = _queue.Queue()
        q.put(self._make_frame())
        q.put(None)

        mock_client = MagicMock()
        mock_submission = MagicMock()
        mock_submission.job_id = "job1"
        mock_client.submit.return_value = mock_submission
        mock_client.wait_for_result.return_value = None

        _classify_worker(q, mock_client)

        assert "Error: classification timed out or failed" in capsys.readouterr().err

    def test_exception_prints_error(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        import queue as _queue

        q: _queue.Queue[NDArray[np.uint8] | None] = _queue.Queue()
        q.put(self._make_frame())
        q.put(None)

        mock_client = MagicMock()
        mock_client.submit.side_effect = RuntimeError("network error")

        _classify_worker(q, mock_client)

        assert "Error during classification: network error" in capsys.readouterr().err

    def test_sentinel_stops_worker(self) -> None:
        import queue as _queue

        q: _queue.Queue[NDArray[np.uint8] | None] = _queue.Queue()
        q.put(None)

        mock_client = MagicMock()
        _classify_worker(q, mock_client)

        mock_client.submit.assert_not_called()
