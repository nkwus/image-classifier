"""Image classifier — webcam capture, food classification, and image saving."""

from image_classifier.annotate import annotate_image
from image_classifier.camera import capture_from_webcam
from image_classifier.capture import DEFAULT_CAPTURES_DIR, save_frame
from image_classifier.classify import classify_food

__all__ = [
    "DEFAULT_CAPTURES_DIR",
    "annotate_image",
    "capture_from_webcam",
    "classify_food",
    "save_frame",
]
