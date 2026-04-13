"""Live webcam preview with frame-saving support."""

from __future__ import annotations

import os
import queue
import sys
import threading

import cv2
import numpy as np
from numpy.typing import NDArray

from image_classifier.api_client import ClassifierClient
from image_classifier.capture import save_frame

WINDOW_NAME = "Webcam - press s=save  q=quit"

_SENTINEL = None  # Pushed to signal the worker thread to exit.

_DEFAULT_API_URL = "http://127.0.0.1:8000"


def _classify_worker(
    q: queue.Queue[NDArray[np.uint8] | None],
    client: ClassifierClient,
) -> None:
    """Background thread that submits frames to the API and saves results.

    Runs until it dequeues the ``_SENTINEL`` value.
    """
    while True:
        frame = q.get()
        if frame is _SENTINEL:
            break
        try:
            # Encode the frame as PNG bytes for submission.
            ok, encoded = cv2.imencode(".png", frame)
            if not ok:
                print("Error: could not encode frame", file=sys.stderr)
                continue
            image_bytes: bytes = encoded.tobytes()

            submission = client.submit(image_bytes)
            job_id = submission.job_id
            print(f"Job {job_id}: submitted, waiting for result…")

            result = client.wait_for_result(job_id, timeout=120.0)
            if result is None:
                print("Error: classification timed out or failed", file=sys.stderr)
                continue

            annotated_bytes, job = result
            # Decode the annotated PNG back to a numpy array for saving.
            buf = np.frombuffer(annotated_bytes, dtype=np.uint8)
            annotated = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if annotated is None:
                print("Error: could not decode annotated image", file=sys.stderr)
                continue

            output_path = save_frame(annotated)
            label = job.label or "Unknown"
            confidence = job.confidence or 0.0
            if output_path is not None:
                print(f"Saved: {output_path} — {label} ({confidence:.1%})")
            else:
                print("Error: could not write capture", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"Error during classification: {exc}", file=sys.stderr)


def capture_from_webcam(
    device: int = 0,
    api_url: str | None = None,
) -> None:
    """Open a live webcam preview window.

    Press **s** to enqueue the current frame for classification (runs in
    a background thread via the API), **q** to quit.
    """
    if api_url is None:
        api_url = os.environ.get("CLASSIFIER_API_URL", _DEFAULT_API_URL)

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(f"Error: could not open camera (device {device}).", file=sys.stderr)
        sys.exit(1)

    print("Webcam open. Press 's' to save a PNG, 'q' to quit.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)

    client = ClassifierClient(api_url)
    classify_queue: queue.Queue[NDArray[np.uint8] | None] = queue.Queue()
    worker = threading.Thread(
        target=_classify_worker, args=(classify_queue, client), daemon=True,
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
        client.close()
        cap.release()
        cv2.destroyAllWindows()
