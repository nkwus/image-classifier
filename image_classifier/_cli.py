"""CLI entry point for the ``image-classifier`` console script."""

from __future__ import annotations

from image_classifier import _qt_fixups


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()

    import argparse

    parser = argparse.ArgumentParser(
        prog="image-classifier",
        description="Webcam food classifier with zero-shot image classification.",
    )
    parser.add_argument(
        "--fetch-labels",
        action="store_true",
        help="Fetch food labels from the USDA FoodData Central API and cache them.",
    )
    args = parser.parse_args()

    if args.fetch_labels:
        from image_classifier.usda import fetch_and_cache_labels

        fetch_and_cache_labels()
        return

    _qt_fixups.apply()

    from image_classifier.camera import capture_from_webcam

    capture_from_webcam()
