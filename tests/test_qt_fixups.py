from __future__ import annotations

import os
from unittest.mock import patch

from image_classifier._qt_fixups import apply


class TestQtFixups:
    """Tests for _qt_fixups.apply()."""

    def test_sets_xcb_platform(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            apply()
            assert os.environ["QT_QPA_PLATFORM"] == "xcb"

    def test_overrides_existing_wayland(self) -> None:
        with patch.dict(os.environ, {"QT_QPA_PLATFORM": "wayland"}):
            apply()
            assert os.environ["QT_QPA_PLATFORM"] == "xcb"
