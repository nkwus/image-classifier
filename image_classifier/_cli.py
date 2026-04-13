"""CLI entry point for the ``image-classifier`` console script."""

from image_classifier import _qt_fixups


def main() -> None:
    _qt_fixups.apply()

    from image_classifier.camera import capture_from_webcam

    capture_from_webcam()
