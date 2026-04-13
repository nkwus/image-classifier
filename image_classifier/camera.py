"""Live webcam preview with frame-saving support."""

import sys

import cv2
import numpy as np
from numpy.typing import NDArray

from image_classifier.capture import save_frame

WINDOW_NAME = "Webcam - press s=save  q=quit"


def capture_from_webcam(device: int = 0) -> None:
    """Open a live webcam preview window.

    Press **s** to save the current frame as a PNG, **q** to quit.
    """
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(f"Error: could not open camera (device {device}).", file=sys.stderr)
        sys.exit(1)

    print("Webcam open. Press 's' to save a PNG, 'q' to quit.")

    # WINDOW_GUI_NORMAL avoids the Qt-based highgui rendering path, which can
    # produce a black preview under XWayland.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)

    try:
        while True:
            ret: bool
            frame: NDArray[np.uint8]
            ret, frame = cap.read()
            if not ret:
                print("Error: failed to read frame from camera.", file=sys.stderr)
                break

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                output_path = save_frame(frame)
                if output_path is not None:
                    print(f"Saved: {output_path}")
                else:
                    print("Error: could not write capture", file=sys.stderr)
    finally:
        cap.release()
        cv2.destroyAllWindows()
