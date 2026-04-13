"""Qt environment workarounds for the opencv-python wheel.

Import this module *before* ``import cv2`` to apply the fixes.
"""

import os


def apply() -> None:
    """Configure Qt environment variables for opencv-python on Linux/Wayland."""
    # The opencv-python wheel bundles only the XCB (X11) Qt platform plugin.
    # On Wayland desktops the shell typically exports QT_QPA_PLATFORM=wayland,
    # which makes Qt fail because the Wayland plugin is missing.  Force XCB so
    # the preview window renders via XWayland.
    os.environ["QT_QPA_PLATFORM"] = "xcb"
