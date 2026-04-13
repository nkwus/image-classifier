from __future__ import annotations

from unittest.mock import MagicMock, patch

from image_classifier._cli import main


class TestCli:
    """Tests for the CLI entry point."""

    def test_main_applies_fixups_and_launches_camera(self) -> None:
        with (
            patch("image_classifier._cli._qt_fixups") as mock_fixups,
            patch("image_classifier.camera.capture_from_webcam") as mock_cam,
        ):
            main()

        mock_fixups.apply.assert_called_once()
        mock_cam.assert_called_once()
