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
    subparsers = parser.add_subparsers(dest="command")

    parser.add_argument(
        "--fetch-labels",
        action="store_true",
        help="Fetch food labels from the USDA FoodData Central API and cache them.",
    )

    serve_parser = subparsers.add_parser("serve", help="Start the classification API server.")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")

    args = parser.parse_args()

    if args.fetch_labels:
        from image_classifier.usda import fetch_and_cache_labels

        fetch_and_cache_labels()
        return

    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "image_classifier.server.app:app",
            host=args.host,
            port=args.port,
        )
        return

    _qt_fixups.apply()

    from image_classifier.camera import capture_from_webcam

    capture_from_webcam()
