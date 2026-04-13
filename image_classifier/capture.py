"""Save camera frames to disk as PNG images."""

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

DEFAULT_CAPTURES_DIR = Path("captures")


def save_frame(
    frame: NDArray[np.uint8],
    output_dir: Path = DEFAULT_CAPTURES_DIR,
) -> Path | None:
    """Save *frame* as a timestamped PNG inside *output_dir*.

    The directory is created automatically if it does not exist.

    Returns the written path on success or ``None`` on failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"capture_{timestamp}.png"
    if cv2.imwrite(str(output_path), frame):
        return output_path
    return None
