from __future__ import annotations

from unittest.mock import MagicMock, patch

from image_classifier._cli import main


class TestCli:
    """Tests for the CLI entry point."""

    def test_main_applies_fixups_and_launches_camera(self) -> None:
        with (
            patch("dotenv.load_dotenv"),
            patch("sys.argv", ["image-classifier"]),
            patch("image_classifier._cli._qt_fixups") as mock_fixups,
            patch("image_classifier.camera.capture_from_webcam") as mock_cam,
        ):
            main()

        mock_fixups.apply.assert_called_once()
        mock_cam.assert_called_once()

    def test_fetch_labels_flag(self) -> None:
        mock_fetch = MagicMock(return_value=["a", "b"])
        with (
            patch("dotenv.load_dotenv"),
            patch("sys.argv", ["image-classifier", "--fetch-labels"]),
            patch(
                "image_classifier.usda.fetch_and_cache_labels", mock_fetch
            ),
        ):
            main()

        mock_fetch.assert_called_once()

    def test_loads_dotenv(self) -> None:
        with (
            patch("dotenv.load_dotenv") as mock_dotenv,
            patch("sys.argv", ["image-classifier"]),
            patch("image_classifier._cli._qt_fixups"),
            patch("image_classifier.camera.capture_from_webcam"),
        ):
            main()

        mock_dotenv.assert_called_once()
