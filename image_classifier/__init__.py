"""Image classifier — webcam capture and image saving utilities."""

from image_classifier.camera import capture_from_webcam
from image_classifier.capture import DEFAULT_CAPTURES_DIR, save_frame

__all__ = [
    "DEFAULT_CAPTURES_DIR",
    "capture_from_webcam",
    "save_frame",
]
