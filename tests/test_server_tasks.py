"""Tests for background classification tasks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import redis as redis_mod

from image_classifier.server.jobs import create_job, get_job, get_result_image
from image_classifier.server.tasks import run_classification

_TASKS_MODULE = "image_classifier.server.tasks"


def _make_png_bytes() -> bytes:
    """Return valid PNG bytes for a small test image."""
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


class TestRunClassification:
    def test_successful_classification(
        self, fake_redis: redis_mod.Redis[bytes],
    ) -> None:
        create_job(fake_redis, "job1")
        image_bytes = _make_png_bytes()

        with (
            patch(f"{_TASKS_MODULE}.classify_food", return_value=("Pizza", 0.95)),
            patch(f"{_TASKS_MODULE}.annotate_image") as mock_annotate,
        ):
            # annotate_image returns a valid frame
            mock_annotate.return_value = np.zeros((48, 64, 3), dtype=np.uint8)
            run_classification("job1", image_bytes, fake_redis)

        data = get_job(fake_redis, "job1")
        assert data is not None
        assert data["status"] == "completed"
        assert data["label"] == "Pizza"
        assert data["confidence"] == 0.95
        assert data["progress_pct"] == 100

        img_data = get_result_image(fake_redis, "job1")
        assert img_data is not None
        assert len(img_data) > 0

    def test_progress_updates(
        self, fake_redis: redis_mod.Redis[bytes],
    ) -> None:
        """Verify that progress is updated to running/5 before inference."""
        create_job(fake_redis, "job1")
        image_bytes = _make_png_bytes()
        snapshots: list[dict[str, object]] = []

        original_classify = MagicMock(return_value=("Sushi", 0.8))

        def capturing_classify(*args: object, **kwargs: object) -> tuple[str, float]:
            # Capture job state right when classify is called
            data = get_job(fake_redis, "job1")
            snapshots.append(data or {})
            return original_classify(*args, **kwargs)

        with (
            patch(f"{_TASKS_MODULE}.classify_food", side_effect=capturing_classify),
            patch(
                f"{_TASKS_MODULE}.annotate_image",
                return_value=np.zeros((48, 64, 3), dtype=np.uint8),
            ),
        ):
            run_classification("job1", image_bytes, fake_redis)

        assert len(snapshots) == 1
        assert snapshots[0]["status"] == "running"
        assert snapshots[0]["progress_pct"] == 10

    def test_records_eta_duration(
        self, fake_redis: redis_mod.Redis[bytes],
    ) -> None:
        create_job(fake_redis, "job1")
        image_bytes = _make_png_bytes()

        with (
            patch(f"{_TASKS_MODULE}.classify_food", return_value=("Tacos", 0.9)),
            patch(
                f"{_TASKS_MODULE}.annotate_image",
                return_value=np.zeros((48, 64, 3), dtype=np.uint8),
            ),
        ):
            run_classification("job1", image_bytes, fake_redis)

        raw = fake_redis.get("stats:avg_duration")
        assert raw is not None
        avg = float(raw)
        assert avg >= 0.0

    def test_failure_sets_failed_status(
        self, fake_redis: redis_mod.Redis[bytes],
    ) -> None:
        create_job(fake_redis, "job1")
        image_bytes = _make_png_bytes()

        with patch(
            f"{_TASKS_MODULE}.classify_food",
            side_effect=RuntimeError("model exploded"),
        ):
            try:
                run_classification("job1", image_bytes, fake_redis)
            except RuntimeError:
                pass

        data = get_job(fake_redis, "job1")
        assert data is not None
        assert data["status"] == "failed"

    def test_invalid_image_bytes(
        self, fake_redis: redis_mod.Redis[bytes],
    ) -> None:
        create_job(fake_redis, "job1")
        run_classification("job1", b"not-an-image", fake_redis)

        data = get_job(fake_redis, "job1")
        assert data is not None
        assert data["status"] == "failed"
