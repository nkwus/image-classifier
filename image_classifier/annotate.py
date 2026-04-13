"""Draw classification results onto an image."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 1.0
_THICKNESS = 2
_PADDING = 12
_LINE_SPACING = 8
_TEXT_COLOR = (255, 255, 255)  # white
_BG_COLOR = (0, 0, 0)  # black


def _wrap_text(text: str, max_width: int) -> list[str]:
    """Split *text* into one or two lines that fit within *max_width*.

    Words are moved to a second line when the full string exceeds
    *max_width*.  If even a single word is wider than *max_width* the
    text is returned as-is on one line (OpenCV will clip it).
    """
    (full_w, _), _ = cv2.getTextSize(text, _FONT, _FONT_SCALE, _THICKNESS)
    if full_w + 2 * _PADDING <= max_width:
        return [text]

    words = text.split()
    line1 = ""
    for i, word in enumerate(words):
        candidate = f"{line1} {word}".strip()
        (cw, _), _ = cv2.getTextSize(candidate, _FONT, _FONT_SCALE, _THICKNESS)
        if cw + 2 * _PADDING > max_width and line1:
            return [line1, " ".join(words[i:])]
        line1 = candidate

    # Everything ended up on one line (single very long word).
    return [text]


def annotate_image(
    image: NDArray[np.uint8],
    label: str,
    confidence: float,
) -> NDArray[np.uint8]:
    """Return a copy of *image* with *label* and *confidence* drawn at the bottom.

    Long captions are wrapped onto two lines so the font stays readable.
    The original array is not modified.
    """
    annotated: NDArray[np.uint8] = image.copy()
    text = f"{label} {confidence:.1%}"
    img_h, img_w = annotated.shape[:2]

    lines = _wrap_text(text, img_w)

    # Measure each line to compute the background box.
    metrics: list[tuple[int, int, int]] = []
    box_w = 0
    total_text_h = 0
    for line in lines:
        (tw, th), bl = cv2.getTextSize(line, _FONT, _FONT_SCALE, _THICKNESS)
        metrics.append((tw, th, bl))
        box_w = max(box_w, tw)
        total_text_h += th + bl

    line_gap = _LINE_SPACING * (len(lines) - 1)
    box_h = total_text_h + 2 * _PADDING + line_gap

    # Solid background rectangle at the bottom.
    cv2.rectangle(
        annotated,
        (0, img_h - box_h),
        (box_w + 2 * _PADDING, img_h),
        _BG_COLOR,
        cv2.FILLED,
    )

    # Draw lines bottom-up so the last line sits at the very bottom.
    y = img_h - _PADDING
    for line, (tw, th, bl) in reversed(list(zip(lines, metrics))):
        cv2.putText(
            annotated,
            line,
            (_PADDING, y - bl),
            _FONT,
            _FONT_SCALE,
            _TEXT_COLOR,
            _THICKNESS,
            cv2.LINE_AA,
        )
        y -= th + bl + _LINE_SPACING

    return annotated
