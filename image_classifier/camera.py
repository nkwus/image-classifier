"""Live webcam preview with frame-saving support."""

from __future__ import annotations

import queue
import sys
import threading

import cv2
import numpy as np
from numpy.typing import NDArray

from image_classifier.annotate import annotate_image
from image_classifier.capture import save_frame
from image_classifier.classify import classify_food

WINDOW_NAME = "Webcam - press s=save  q=quit"

_SENTINEL = None  # Pushed to signal the worker thread to exit.


def _classify_worker(
    q: queue.Queue[NDArray[np.uint8] | None],
) -> None:
    """Background thread that classifies and saves frames from *q*.

    Runs until it dequeues the ``_SENTINEL`` value.
    """
    while True:
        frame = q.get()
        if frame is _SENTINEL:
            break
        try:
            label, confidence = classify_food(frame)
            annotated = annotate_image(frame, label, confidence)
            output_path = save_frame(annotated)
            if output_path is not None:
                print(f"Saved: {output_path} — {label} ({confidence:.1%})")
            else:
                print("Error: could not write capture", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"Error during classification: {exc}", file=sys.stderr)


def capture_from_webcam(device: int = 0) -> None:
    """Open a live webcam preview window.

    Press **s** to enqueue the current frame for classification (runs in
    a background thread), **q** to quit.
    """
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(f"Error: could not open camera (device {device}).", file=sys.stderr)
        sys.exit(1)

    print("Webcam open. Press 's' to save a PNG, 'q' to quit.")

    # WINDOW_GUI_NORMAL avoids the Qt-based highgui rendering path, which can
    # produce a black preview under XWayland.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)

    classify_queue: queue.Queue[NDArray[np.uint8] | None] = queue.Queue()
    worker = threading.Thread(
        target=_classify_worker, args=(classify_queue,), daemon=True,
    )
    worker.start()

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
                classify_queue.put(frame.copy())
                print("Queued frame for classification…")
    finally:
        classify_queue.put(_SENTINEL)
        worker.join()
        cap.release()
        cv2.destroyAllWindows()
