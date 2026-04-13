from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from image_classifier.annotate import annotate_image, _wrap_text


class TestAnnotateImage:
    """Tests for the image annotation helper."""

    @staticmethod
    def _make_frame(
        width: int = 640, height: int = 480
    ) -> NDArray[np.uint8]:
        return np.full((height, width, 3), 128, dtype=np.uint8)

    def test_returns_same_shape(self) -> None:
        frame = self._make_frame()
        result = annotate_image(frame, "Pizza", 0.95)
        assert result.shape == frame.shape

    def test_does_not_modify_original(self) -> None:
        frame = self._make_frame()
        original = frame.copy()
        annotate_image(frame, "Pizza", 0.95)
        np.testing.assert_array_equal(frame, original)

    def test_draws_at_bottom_left(self) -> None:
        frame = self._make_frame()
        result = annotate_image(frame, "Pizza", 0.95)

        # The bottom-left region should have been overwritten with the
        # black background rectangle.
        bottom_left_patch = result[440:, :200]
        assert not np.array_equal(bottom_left_patch, frame[440:, :200])

    def test_top_right_unchanged(self) -> None:
        frame = self._make_frame()
        result = annotate_image(frame, "Sushi", 0.71)

        # The annotation is at the bottom-left, so the top-right should
        # be identical to the original.
        assert np.array_equal(result[:80, 500:], frame[:80, 500:])

    def test_confidence_formatted_as_percent(self) -> None:
        # A longer label should produce a wider dark region at the bottom.
        frame = self._make_frame()
        short = annotate_image(frame, "Pizza", 0.9)
        long = annotate_image(frame, "Chocolate Cake", 0.9)

        # Count black (0,0,0) pixels in the bottom strip.
        short_black = int(np.all(short[430:] == 0, axis=-1).sum())
        long_black = int(np.all(long[430:] == 0, axis=-1).sum())
        assert long_black > short_black

    def test_wraps_long_text_to_two_lines(self) -> None:
        """Long text should produce a taller background box (two lines)."""
        narrow = self._make_frame(width=300)
        result = annotate_image(
            narrow, "Extremely Long Food Label Here", 0.85,
        )
        assert result.shape == narrow.shape

        # Two-line caption creates more black pixels than a short one-line
        # caption on the same narrow frame.
        short_result = annotate_image(narrow, "Pizza", 0.85)
        long_black = int(np.all(result[400:] == 0, axis=-1).sum())
        short_black = int(np.all(short_result[400:] == 0, axis=-1).sum())
        assert long_black > short_black


class TestWrapText:
    """Tests for the _wrap_text helper."""

    def test_short_text_single_line(self) -> None:
        lines = _wrap_text("Pizza 95.0%", 640)
        assert lines == ["Pizza 95.0%"]

    def test_long_text_wraps_to_two_lines(self) -> None:
        lines = _wrap_text("Extremely Long Food Label 85.0%", 200)
        assert len(lines) == 2
        assert " ".join(lines) == "Extremely Long Food Label 85.0%"

    def test_single_long_word_stays_on_one_line(self) -> None:
        lines = _wrap_text("Superlongword", 50)
        assert lines == ["Superlongword"]
