from __future__ import annotations

from unittest.mock import patch


class TestMainModule:
    """Tests for ``python -m image_classifier``."""

    def test_invokes_cli_main(self) -> None:
        with patch("image_classifier._cli.main") as mock_main:
            import importlib
            import image_classifier.__main__ as mod

            importlib.reload(mod)

        mock_main.assert_called()
