# image-classifier

Webcam capture tool that opens a live camera preview and lets you save frames as PNG images. Built with OpenCV and packaged as an installable Python library so it can be imported into other projects or run from the command line.

## Prerequisites

- **Python 3.12+**
- **A webcam** accessible at `/dev/video0` (the default V4L2 device on Linux)
- **[uv](https://docs.astral.sh/uv/)** (recommended) or pip

### Linux display server note

This project runs on both X11 and Wayland desktops. On Wayland the CLI automatically forces X11/XCB mode via XWayland because the `opencv-python` wheel does not bundle a Wayland Qt plugin. See [_qt_fixups.py](#qt--xwayland-workarounds) below for details.

## Setup

```bash
# Clone the repo
git clone <repo-url> && cd image-classifier

# Install dependencies (creates .venv automatically)
uv sync
```

If the OpenCV preview window fonts produce warnings about a missing font directory, symlink system fonts into the wheel's expected path:

```bash
rmdir .venv/lib/python3.12/site-packages/cv2/qt/fonts
ln -s /usr/share/fonts/truetype .venv/lib/python3.12/site-packages/cv2/qt/fonts
```

## Usage

### Command line

```bash
# Either of these launches the webcam preview window:
uv run image-classifier
python -m image_classifier
```

| Key | Action                                                       |
| --- | ------------------------------------------------------------ |
| `s` | Save the current frame as a PNG to the `captures/` directory |
| `q` | Quit the preview and release the camera                      |

Saved files are named `capture_YYYYMMDD_HHMMSS.png` (e.g. `capture_20260412_153045.png`). The `captures/` directory is created automatically on first save and is git-ignored.

### As a library

```python
from pathlib import Path

import cv2
from image_classifier import save_frame

# Save an arbitrary frame to a custom directory
cap = cv2.VideoCapture(0)
_, frame = cap.read()
path = save_frame(frame, output_dir=Path("my_images"))
print(path)  # my_images/capture_20260412_153045.png
cap.release()
```

The public API exported from `image_classifier` is:

| Symbol                              | Type     | Description                                    |
| ----------------------------------- | -------- | ---------------------------------------------- |
| `save_frame(frame, output_dir=...)` | function | Save a numpy frame as a timestamped PNG        |
| `capture_from_webcam(device=0)`     | function | Open a live preview window with save/quit keys |
| `DEFAULT_CAPTURES_DIR`              | `Path`   | Default output directory (`captures/`)         |

> **Important:** When using `capture_from_webcam` from your own code, call `image_classifier._qt_fixups.apply()` before import if you are on Wayland. The CLI handles this automatically, but the library intentionally does not override your Qt environment on import.

## Project structure

```
image_classifier/
    __init__.py       Public API exports
    __main__.py       python -m entry point
    _cli.py           Console script entry point
    _qt_fixups.py     Qt/XWayland environment workarounds
    camera.py         Webcam preview loop
    capture.py        Frame-to-PNG saving logic
tests/
    test_camera.py        Webcam loop tests (camera and UI fully mocked)
    test_capture.py       save_frame tests (real filesystem via tmp_path)
    test_cli.py           CLI entry point test
    test_main_module.py   python -m entry point test
    test_qt_fixups.py     Qt fixup tests
pyproject.toml            Package metadata, dependencies, build config
```

## Design decisions

### Why split into multiple modules?

Each module has a single responsibility:

- **`capture.py`** handles saving a numpy array to disk as PNG. It knows nothing about cameras or windows. This makes it easy to test (no mocking of UI) and reusable in any pipeline that produces frames.
- **`camera.py`** handles the webcam read loop and the OpenCV preview window. It delegates saving to `capture.save_frame`.
- **`_qt_fixups.py`** contains environment variable overrides needed to make OpenCV's bundled Qt work on Wayland/XWayland. It is isolated in its own module so that library consumers are not affected — the fixups are only applied at CLI entry points (`_cli.py` and `__main__.py`), not on `import image_classifier`.
- **`_cli.py`** and **`__main__.py`** are thin wrappers that apply the Qt fixups and then call the camera function. The underscore prefix on `_cli` and `_qt_fixups` signals these are internal implementation details, not part of the public API.

### Why `WINDOW_GUI_NORMAL`?

OpenCV's default `imshow` uses a Qt-based rendering path (`WINDOW_GUI_EXPANDED`) that produces a black preview window under XWayland. `WINDOW_GUI_NORMAL` forces the classic rendering path which works reliably.

### Qt / XWayland workarounds

The `opencv-python` PyPI wheel bundles its own Qt build, but only includes the XCB (X11) platform plugin — no Wayland plugin. On Wayland desktops the shell usually exports `QT_QPA_PLATFORM=wayland`, which makes Qt fail because the plugin is missing.

`_qt_fixups.py` force-sets `QT_QPA_PLATFORM=xcb` before OpenCV is imported. This routes the window through XWayland, which is available on all modern Wayland compositors.

The font directory symlink (see [Setup](#setup)) is a separate issue: Qt looks for fonts at a hardcoded path inside the wheel that ships empty.

### Why timestamped filenames?

Filenames like `capture_20260412_153045.png` avoid overwriting previous captures. The timestamp is second-precision, which is sufficient for interactive use. If sub-second captures were needed, the format could be extended with `%f` (microseconds).

### Why `Path | None` return instead of raising?

`save_frame` returns `None` on write failure rather than raising an exception. This keeps the camera loop simple — a failed save should not crash the preview. The caller decides how to handle the failure (the CLI prints to stderr and continues).

### Testing strategy

- **`test_capture.py`** tests `save_frame` against the real filesystem using pytest's `tmp_path` fixture. No mocking except for the failure case (`cv2.imwrite` patched to return `False`).
- **`test_camera.py`** tests the webcam loop with `cv2.VideoCapture`, `cv2.imshow`, `cv2.waitKey` etc. fully mocked via `unittest.mock.patch`. This lets the tests run in CI without a camera or display. Each test verifies one behaviour: camera-open failure, quit key, save key, save failure, and read failure.
- **`test_qt_fixups.py`** verifies that `apply()` sets `QT_QPA_PLATFORM=xcb` and overrides an existing Wayland setting.
- **`test_cli.py`** verifies the CLI entry point applies Qt fixups before launching the camera.
- **`test_main_module.py`** verifies that `python -m image_classifier` delegates to the CLI entry point.

All 14 tests achieve **100% line coverage** across the package.

## Running tests

```bash
# Run tests
uv run pytest tests/ -v

# Run tests with coverage report
uv run pytest tests/ --cov=image_classifier --cov-report=term-missing -v
```