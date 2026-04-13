from __future__ import annotations

from unittest.mock import MagicMock, patch

from image_classifier._cli import main


class TestCli:
    """Tests for the CLI entry point."""

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
        mock_uvicorn = MagicMock()
        with (
            patch("dotenv.load_dotenv") as mock_dotenv,
            patch("sys.argv", ["image-classifier"]),
            patch("uvicorn.run", mock_uvicorn),
        ):
            main()

        mock_dotenv.assert_called_once()

    def test_default_starts_server(self) -> None:
        mock_uvicorn = MagicMock()
        with (
            patch("dotenv.load_dotenv"),
            patch("sys.argv", ["image-classifier"]),
            patch("uvicorn.run", mock_uvicorn),
        ):
            main()

        mock_uvicorn.assert_called_once_with(
            "image_classifier.server.app:app",
            host="0.0.0.0",
            port=8000,
        )

    def test_custom_host_port(self) -> None:
        mock_uvicorn = MagicMock()
        with (
            patch("dotenv.load_dotenv"),
            patch("sys.argv", ["image-classifier", "--host", "127.0.0.1", "--port", "9000"]),
            patch("uvicorn.run", mock_uvicorn),
        ):
            main()

        mock_uvicorn.assert_called_once_with(
            "image_classifier.server.app:app",
            host="127.0.0.1",
            port=9000,
        )
