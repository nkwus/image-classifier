# image-classifier

Webcam capture tool that opens a live camera preview, classifies food in the frame using zero-shot image classification ([google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) — SigLIP vision-language model), and saves annotated PNG images with the food label and confidence score overlaid. Candidate labels can be fetched from the [USDA FoodData Central API](https://fdc.nal.usda.gov/api-guide) for broad food coverage, or you can supply your own. Built with OpenCV and HuggingFace Transformers, and packaged as an installable Python library.

## Prerequisites

- **Python 3.12+**
- **A webcam** accessible at `/dev/video0` (the default V4L2 device on Linux)
- **[uv](https://docs.astral.sh/uv/)** (recommended) or pip
- **(Optional) A [USDA FoodData Central API key](https://fdc.nal.usda.gov/api-key-signup)** — required only for fetching an expanded food label list from the USDA

### Linux display server note

This project runs on both X11 and Wayland desktops. On Wayland the CLI automatically forces X11/XCB mode via XWayland because the `opencv-python` wheel does not bundle a Wayland Qt plugin. See [_qt_fixups.py](#qt--xwayland-workarounds) below for details.

## Setup

```bash
# Clone the repo
git clone <repo-url> && cd image-classifier

# Install dependencies (creates .venv automatically)
uv sync
```

Create a `.env` file in the project root for your API keys (this file is git-ignored):

```bash
# .env
FDA_API_KEY=your_usda_api_key_here
```

The CLI loads `.env` automatically via `python-dotenv`. The VS Code workspace is also configured (`python.envFile`) so the Python extension picks up these variables for debugging, testing, and Pylance.

If the OpenCV preview window fonts produce warnings about a missing font directory, symlink system fonts into the wheel's expected path:

```bash
rmdir .venv/lib/python3.12/site-packages/cv2/qt/fonts
ln -s /usr/share/fonts/truetype .venv/lib/python3.12/site-packages/cv2/qt/fonts
```

## Usage

### Command line

```bash
# Launch the webcam preview window:
uv run image-classifier
python -m image_classifier

# Fetch food labels from the USDA FoodData Central API and cache them:
uv run image-classifier --fetch-labels
```

The `--fetch-labels` flag pages through the USDA `/foods/list` endpoint (Foundation and SR Legacy data types), collects all food descriptions, and caches them as JSON in `~/.cache/image-classifier/usda_labels.json`. The API key is read from `FDA_API_KEY` in `.env` (or from the environment). The USDA rate limit is 1,000 requests per hour per API key.

Once cached, the USDA labels are used automatically as the candidate label set for zero-shot classification. If no cache exists, the built-in Food-101 label list is used as a fallback.

| Key | Action                                                                                     |
| --- | ------------------------------------------------------------------------------------------ |
| `s` | Classify the food in the current frame, overlay the result, and save as PNG to `captures/` |
| `q` | Quit the preview and release the camera                                                    |

Saved files are named `capture_YYYYMMDD_HHMMSS.png` (e.g. `capture_20260412_153045.png`). Each saved PNG has the predicted food label and confidence percentage rendered in the top-left corner. The `captures/` directory is created automatically on first save and is git-ignored.

The SigLIP model is downloaded from HuggingFace Hub on first use and cached locally (~3.5 GB).

### As a library

```python
from pathlib import Path

import cv2
from image_classifier import annotate_image, classify_food, save_frame

cap = cv2.VideoCapture(0)
_, frame = cap.read()
cap.release()

# Classify and annotate
label, confidence = classify_food(frame)
annotated = annotate_image(frame, label, confidence)

# Save the annotated image
path = save_frame(annotated, output_dir=Path("my_images"))
print(f"{path} — {label} ({confidence:.1%})")
```

The public API exported from `image_classifier` is:

| Symbol                                     | Type     | Description                                                           |
| ------------------------------------------ | -------- | --------------------------------------------------------------------- |
| `classify_food(image, candidate_labels=…)` | function | Zero-shot classify a BGR image as a food item → `(label, confidence)` |
| `annotate_image(image, label, confidence)` | function | Return a copy with label/confidence drawn in top-left                 |
| `save_frame(frame, output_dir=...)`        | function | Save a numpy frame as a timestamped PNG                               |
| `capture_from_webcam(device=0)`            | function | Open a live preview window with save/quit keys                        |
| `DEFAULT_CAPTURES_DIR`                     | `Path`   | Default output directory (`captures/`)                                |

> **Important:** When using `capture_from_webcam` from your own code, call `image_classifier._qt_fixups.apply()` before import if you are on Wayland. The CLI handles this automatically, but the library intentionally does not override your Qt environment on import.

## Project structure

```
image_classifier/
    __init__.py       Public API exports
    __main__.py       python -m entry point
    _cli.py           Console script entry point
    _qt_fixups.py     Qt/XWayland environment workarounds
    annotate.py       Draw classification results onto an image
    camera.py         Webcam preview loop
    capture.py        Frame-to-PNG saving logic
    classify.py       Food classification using a pretrained model
    usda.py           USDA FoodData Central API client and label cache
tests/
    test_annotate.py      Image annotation tests
    test_camera.py        Webcam loop tests (camera and UI fully mocked)
    test_capture.py       save_frame tests (real filesystem via tmp_path)
    test_classify.py      Food classification tests (model mocked)
    test_cli.py           CLI entry point test
    test_main_module.py   python -m entry point test
    test_qt_fixups.py     Qt fixup tests
    test_usda.py          USDA API client tests (HTTP fully mocked)
pyproject.toml            Package metadata, dependencies, build config
.env                      API keys (git-ignored)
```

## Design decisions

### Why split into multiple modules?

Each module has a single responsibility:

- **`capture.py`** handles saving a numpy array to disk as PNG. It knows nothing about cameras or windows. This makes it easy to test (no mocking of UI) and reusable in any pipeline that produces frames.
- **`classify.py`** wraps a HuggingFace `transformers` zero-shot image-classification pipeline using the SigLIP vision-language model. Instead of a model fine-tuned on fixed classes, the zero-shot approach compares the image embedding against text embeddings for each candidate label — so you can change the label set at runtime without retraining. When no explicit labels are passed, it checks for cached USDA labels (see `usda.py`) and falls back to the built-in Food-101 list. When the candidate list exceeds 100 labels, classification is batched (100 labels per batch) to avoid GPU out-of-memory errors; a `tqdm` progress bar tracks batch progress. The `transformers` library emits a noisy "sequential pipeline on GPU" warning for each batch — this is suppressed by temporarily raising the log level on `transformers.pipelines.base` during the classification loop. SigLIP's tokenizer (`SiglipTokenizer`) uses SentencePiece models serialised with Protocol Buffers, so `sentencepiece` and `protobuf` are both required at runtime. The model is loaded lazily on first use (via `_get_classifier`) and cached for the lifetime of the process. The deferred `from transformers import pipeline` inside `_create_pipeline` keeps `torch` out of the import graph until classification is actually requested, so startup stays fast.
- **`usda.py`** fetches food descriptions from the [USDA FoodData Central API](https://fdc.nal.usda.gov/api-guide) and caches them as a JSON file (`~/.cache/image-classifier/usda_labels.json`). It uses the POST `/v1/foods/list` endpoint, paging through Foundation and SR Legacy data types at 200 items per page (the API maximum). The API key is sent via the `X-Api-Key` HTTP header rather than as a query parameter so it does not leak into URLs or server logs. `load_cached_labels()` is the only function called at classification time — it reads from disk with no network access. The fetching is triggered explicitly by `--fetch-labels`.
- **`annotate.py`** draws a label and confidence string onto the top-left corner of an image using OpenCV's `putText` with a solid black background rectangle for readability. Returns a copy — the original frame is never modified.
- **`camera.py`** handles the webcam read loop and the OpenCV preview window. On save it calls `classify_food` → `annotate_image` → `save_frame` to produce the annotated PNG.
- **`_qt_fixups.py`** contains environment variable overrides needed to make OpenCV's bundled Qt work on Wayland/XWayland. It is isolated in its own module so that library consumers are not affected — the fixups are only applied at CLI entry points (`_cli.py` and `__main__.py`), not on `import image_classifier`.
- **`_cli.py`** and **`__main__.py`** are thin wrappers that load `.env` via `python-dotenv`, parse CLI arguments, apply the Qt fixups, and launch the camera or fetch USDA labels. The underscore prefix on `_cli` and `_qt_fixups` signals these are internal implementation details, not part of the public API.

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

### Why SigLIP zero-shot instead of a fine-tuned model?

A fine-tuned classifier (e.g. a ViT trained on Food-101) can only predict the exact labels it was trained on, and its accuracy is limited by the quality and breadth of the training set. [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384) is a vision-language model trained on image–text pairs: it embeds images and text into a shared space and scores how well each candidate label matches the image. This "zero-shot" approach means you can change the label list at runtime — pass `candidate_labels=["espresso", "latte", "cappuccino"]` to `classify_food` and it just works, with no retraining. The default labels are the 101 Food-101 categories plus coffee, tea, and soda. The trade-off is model size (~3.5 GB vs ~330 MB for a fine-tuned ViT) and slightly higher latency per classification. SigLIP's tokenizer (`SiglipTokenizer`) is built on SentencePiece and serialised with Protocol Buffers, which is why `sentencepiece` and `protobuf` are listed as dependencies.

### Why USDA FoodData Central for labels?

The built-in Food-101 list covers ~104 categories — enough for a demo, but users quickly run into foods that aren't listed. The [USDA FoodData Central API](https://fdc.nal.usda.gov/api-guide) publishes thousands of curated food descriptions under the public-domain CC0 licence. By fetching Foundation and SR Legacy data types we get clean, human-readable descriptions (e.g. "strawberries, raw", "cheese, cheddar") that work well as zero-shot candidate labels.

The fetch is explicit (`--fetch-labels`) rather than automatic so the app never makes surprise network calls, and the cached JSON file means classification time has zero API dependency. More labels does mean slower zero-shot scoring — with thousands of candidates each classification takes longer than with the 104-item fallback list. To keep GPU memory under control, `classify_food` processes candidate labels in batches of 100 and keeps only the best-scoring result across all batches. A `tqdm` progress bar is shown during batched classification so users can track progress. For latency-sensitive use cases, pass a focused `candidate_labels` list to `classify_food()`.

### Why `python-dotenv`?

The USDA API key must be kept out of source control. Storing it in a `.env` file (which is git-ignored) and loading it at CLI startup via `python-dotenv` is a standard pattern. The VS Code workspace file also sets `python.envFile` so the Python extension loads the same variables for debugging and testing.

### Why lazy-load the model?

Importing `torch` and loading model weights takes several seconds. By deferring this to the first `s` keypress (via `_get_classifier`), the webcam preview opens instantly. Subsequent saves reuse the cached pipeline with no delay.

### Why overlay text on the saved image?

The classification result is burned into the PNG so the file is self-documenting — you can browse the `captures/` directory and immediately see what each image was classified as, without needing to re-run the model or cross-reference a log file.

### Testing strategy

- **`test_capture.py`** tests `save_frame` against the real filesystem using pytest's `tmp_path` fixture. No mocking except for the failure case (`cv2.imwrite` patched to return `False`).
- **`test_classify.py`** tests `classify_food` with the `_classifier` global injected as a mock — no real model download or GPU needed. Verifies label formatting (title case), PIL conversion, lazy loading, caching, custom `candidate_labels`, USDA cache integration (uses cached USDA labels when available, falls back to `FOOD_LABELS` when not), and label batching (with more labels than `_LABEL_BATCH_SIZE`, multiple batches run and the best score across all batches wins). A separate `TestCreatePipeline` class patches `sys.modules` to verify `_create_pipeline` calls `transformers.pipeline` with the correct model name and `"zero-shot-image-classification"` task. `TestFoodLabels` validates the `FOOD_LABELS` list is non-empty with no duplicates.
- **`test_annotate.py`** tests `annotate_image` at the pixel level: same shape, original not modified, top-left corner changed, bottom-right unchanged, and background rectangle grows with longer labels.
- **`test_camera.py`** tests the webcam loop with `cv2.VideoCapture`, `cv2.imshow`, `cv2.waitKey`, `classify_food`, and `annotate_image` fully mocked via `unittest.mock.patch`. This lets the tests run in CI without a camera, display, or model. Each test verifies one behaviour: camera-open failure, quit key, save key (with classify + annotate), save failure, and read failure.
- **`test_qt_fixups.py`** verifies that `apply()` sets `QT_QPA_PLATFORM=xcb` and overrides an existing Wayland setting.
- **`test_cli.py`** verifies the CLI entry point loads `.env`, applies Qt fixups before launching the camera, and handles the `--fetch-labels` flag.
- **`test_main_module.py`** verifies that `python -m image_classifier` delegates to the CLI entry point.
- **`test_usda.py`** tests the USDA API client with HTTP requests fully mocked (no real network calls). Verifies request headers (`X-Api-Key`, `Content-Type`), pagination, deduplication, cache round-tripping, invalid cache handling, and missing API key behaviour.

All 51 tests achieve **100% line coverage** (206 statements) across the package.

## Running tests

```bash
# Run tests
uv run pytest tests/ -v

# Run tests with coverage report
uv run pytest tests/ --cov=image_classifier --cov-report=term-missing -v
```