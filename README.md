# image-classifier

Food classification REST API that accepts images and classifies them using zero-shot image classification ([google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) — SigLIP vision-language model). Returns annotated PNG images with the food label and confidence score overlaid. Candidate labels can be fetched from the [USDA FoodData Central API](https://fdc.nal.usda.gov/api-guide) for broad food coverage. Built with FastAPI, OpenCV, Redis, and HuggingFace Transformers, designed to run in a Docker container.

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or pip
- **A [Redis](https://redis.io/) instance** — the API server uses Redis to store job state and result images (a [Redis Cloud](https://redis.io/cloud/) free tier works fine)
- **(Optional) A [USDA FoodData Central API key](https://fdc.nal.usda.gov/api-key-signup)** — required only for fetching an expanded food label list from the USDA

## Setup

```bash
# Clone the repo
git clone <repo-url> && cd image-classifier

# Install dependencies (creates .venv automatically)
uv sync
```

Create a `.env` file in the project root for your API keys and Redis credentials (this file is git-ignored):

```bash
# .env
FDA_API_KEY=your_usda_api_key_here
HF_TOKEN=your_huggingface_token_here
REDIS_URL=redis://your-redis-host:port/0
REDIS_USERNAME=your_redis_username
REDIS_PASSWORD=your_redis_password
```

The `REDIS_URL`, `REDIS_USERNAME`, and `REDIS_PASSWORD` variables are required for the API server. `FDA_API_KEY` is only needed for the `--fetch-labels` command. `HF_TOKEN` enables authenticated HuggingFace Hub downloads (optional but recommended for higher rate limits).

The CLI loads `.env` automatically via `python-dotenv`. The VS Code workspace is also configured (`python.envFile`) so the Python extension picks up these variables for debugging, testing, and Pylance.

## Usage

### API

The server exposes a REST API using an async job queue pattern:

1. Submit an image to `POST /classify` → receive a job ID (202 Accepted)
2. Poll `GET /jobs/{job_id}` until the job completes (with progress % and ETA)
3. Download the annotated image from `GET /jobs/{job_id}/image`

### Command line

```bash
# Start the classification API server (default: 0.0.0.0:8000):
uv run image-classifier
uv run image-classifier --host 127.0.0.1 --port 9000

# Fetch food labels from the USDA FoodData Central API and cache them:
uv run image-classifier --fetch-labels
```

The `--fetch-labels` flag pages through the USDA `/foods/list` endpoint (Foundation and SR Legacy data types), collects all food descriptions, and caches them as JSON in `~/.cache/image-classifier/usda_labels.json`. The API key is read from `FDA_API_KEY` in `.env` (or from the environment). The USDA rate limit is 1,000 requests per hour per API key.

Once cached, the USDA labels are used automatically as the candidate label set for zero-shot classification. If no cache exists, the built-in Food-101 label list is used as a fallback.

The SigLIP model is downloaded from HuggingFace Hub on first use and cached locally (~3.5 GB).

### Docker

```bash
docker build -t image-classifier .
docker run -p 8000:8000 --env-file .env image-classifier
```

### As a library

```python
import cv2
from image_classifier import annotate_image, classify_food

frame = cv2.imread("photo.jpg")

label, confidence = classify_food(frame)
annotated = annotate_image(frame, label, confidence)

cv2.imwrite("result.png", annotated)
print(f"{label} ({confidence:.1%})")
```

The public API exported from `image_classifier` is:

| Symbol                                     | Type     | Description                                                           |
| ------------------------------------------ | -------- | --------------------------------------------------------------------- |
| `classify_food(image, candidate_labels=…)` | function | Zero-shot classify a BGR image as a food item → `(label, confidence)` |
| `annotate_image(image, label, confidence)` | function | Return a copy with label/confidence drawn in top-left                 |

## Project structure

```
image_classifier/
    __init__.py       Public API exports
    _cli.py           Console script entry point (server + fetch-labels)
    annotate.py       Draw classification results onto an image
    classify.py       Food classification using a pretrained model
    usda.py           USDA FoodData Central API client and label cache
    server/
        __init__.py   Package marker
        app.py        FastAPI instance and lifespan (Redis + executor setup)
        models.py     Server-side Pydantic models
        jobs.py       Redis-backed job CRUD (create, update, get, delete)
        tasks.py      Background classification orchestration
        routes.py     HTTP route handlers (POST /classify, GET /jobs, etc.)
tests/
    conftest.py           Shared test fixtures (fakeredis)
    test_annotate.py      Image annotation tests
    test_classify.py      Food classification tests (model mocked)
    test_cli.py           CLI entry point tests
    test_server_jobs.py   Redis job store tests (fakeredis)
    test_server_models.py Server-side Pydantic model validation tests
    test_server_routes.py FastAPI route integration tests
    test_server_tasks.py  Background task tests
    test_usda.py          USDA API client tests (HTTP fully mocked)
Dockerfile                Container image definition
pyproject.toml            Package metadata, dependencies, build config
.env                      API keys and Redis credentials (git-ignored)
```

## How classification works

### The model: SigLIP (google/siglip-so400m-patch14-384)

This project uses Google's **SigLIP** (Sigmoid Loss for Language-Image Pre-training) vision-language model, specifically the [`google/siglip-so400m-patch14-384`](https://huggingface.co/google/siglip-so400m-patch14-384) checkpoint hosted on Hugging Face.

SigLIP is a **vision-language model** — it was trained on hundreds of millions of image–text pairs so that it learns a shared embedding space for images and text. Given an image and a list of text strings ("candidate labels"), SigLIP scores how well each label matches the image. This is called **zero-shot image classification** because the model was never explicitly trained on those specific labels — it generalises from its language understanding.

**Why SigLIP instead of a fine-tuned classifier?** A fine-tuned model (e.g. a ViT trained on the Food-101 dataset) can only predict the exact classes it was trained on. If you show it a food that wasn't in its training set, it will confidently pick the closest wrong answer. SigLIP, by contrast, can classify against *any* set of text labels you provide at runtime — no retraining needed. The trade-off is model size (~3.5 GB vs ~330 MB for a fine-tuned ViT) and slightly higher latency.

The model is loaded via the HuggingFace `transformers` library as a `"zero-shot-image-classification"` pipeline (see [classify.py](image_classifier/classify.py#L66)). It is downloaded from Hugging Face Hub on first use and cached locally.

### Why candidate labels are required (and what happens without them)

Zero-shot classification does **not** generate a label from thin air. It **compares** the image against every label in a provided list and returns the best match with a confidence score. Without a list of candidate labels, the model has nothing to compare against and cannot produce any output.

This is fundamentally different from an image captioning model (which generates free-form text) or a fine-tuned classifier (which has labels baked into its final layer). Zero-shot classification is essentially a ranked similarity search: "which of these N text descriptions best matches this image?"

The quality of results depends directly on the candidate labels:
- **Too few labels** — the model is forced to pick from a limited set, so it may assign a confident but wrong label (e.g. if you only provide `["pizza", "sushi"]` and show it a hamburger, it will pick whichever of those two is closest).
- **Too many labels** — accuracy improves (the correct label is more likely to be present) but classification slows down because the model must score every candidate. With thousands of labels, the pipeline batches them (100 per batch) to avoid GPU out-of-memory errors.
- **Missing the correct label** — the model will always return *something* from the list, even if the true food isn't present. It cannot say "none of the above".

### Where the labels come from

1. **Built-in Food-101 list** — [classify.py](image_classifier/classify.py#L19) ships with ~110 hard-coded labels (the standard Food-101 categories plus coffee, tea, soda, and a few extras). This is the fallback when no USDA cache exists.

2. **USDA FoodData Central** — The [USDA FoodData Central API](https://fdc.nal.usda.gov/api-guide) publishes thousands of curated, human-readable food descriptions (e.g. "strawberries, raw", "cheese, cheddar") under a public-domain CC0 licence. Running `uv run image-classifier --fetch-labels` downloads these descriptions and caches them as JSON at `~/.cache/image-classifier/usda_labels.json`. Once cached, they are automatically used as the candidate label set, giving much broader food coverage than the 110-item fallback.

3. **Custom labels at runtime** — You can pass any list you want via the `candidate_labels` parameter to `classify_food()` (e.g. `["espresso", "latte", "cappuccino"]` for a coffee-specific classifier).

## Design decisions

### Why split into multiple modules?

Each module has a single responsibility:

- **`classify.py`** wraps a HuggingFace `transformers` zero-shot image-classification pipeline using the SigLIP vision-language model. Instead of a model fine-tuned on fixed classes, the zero-shot approach compares the image embedding against text embeddings for each candidate label — so you can change the label set at runtime without retraining. When no explicit labels are passed, it checks for cached USDA labels (see `usda.py`) and falls back to the built-in Food-101 list. When the candidate list exceeds 100 labels, classification is batched (100 labels per batch) to avoid GPU out-of-memory errors; a `tqdm` progress bar tracks batch progress. The `transformers` library emits a noisy "sequential pipeline on GPU" warning for each batch — this is suppressed by temporarily raising the log level on `transformers.pipelines.base` during the classification loop. SigLIP's tokenizer (`SiglipTokenizer`) uses SentencePiece models serialised with Protocol Buffers, so `sentencepiece` and `protobuf` are both required at runtime. The model is loaded lazily on first use (via `_get_classifier`) and cached for the lifetime of the process. The deferred `from transformers import pipeline` inside `_create_pipeline` keeps `torch` out of the import graph until classification is actually requested, so startup stays fast.
- **`usda.py`** fetches food descriptions from the [USDA FoodData Central API](https://fdc.nal.usda.gov/api-guide) and caches them as a JSON file (`~/.cache/image-classifier/usda_labels.json`). It uses the POST `/v1/foods/list` endpoint, paging through Foundation and SR Legacy data types at 200 items per page (the API maximum). The API key is sent via the `X-Api-Key` HTTP header rather than as a query parameter so it does not leak into URLs or server logs. `load_cached_labels()` is the only function called at classification time — it reads from disk with no network access. The fetching is triggered explicitly by `--fetch-labels`.
- **`annotate.py`** draws a label and confidence string onto the top-left corner of an image using OpenCV's `putText` with a solid black background rectangle for readability. Returns a copy — the original frame is never modified.
- **`server/app.py`** creates the FastAPI instance with a lifespan hook that connects to Redis, creates a `ThreadPoolExecutor` (single worker for the non-thread-safe SigLIP model), and pre-loads the model.
- **`server/routes.py`** contains thin HTTP route handlers that validate requests, dispatch background tasks, and format responses. All Redis calls are wrapped in `asyncio.to_thread()` to avoid blocking the event loop.
- **`server/jobs.py`** is a synchronous Redis-backed job store. Jobs are stored as Redis hashes with a 1-hour TTL. Result images are stored as separate keys.
- **`server/tasks.py`** orchestrates background classification: decode image → classify → annotate → store result. It updates job progress (10% → 80% → 100%) and tracks an EMA rolling average of inference duration for ETA estimation.
- **`_cli.py`** is a thin wrapper that loads `.env` via `python-dotenv`, parses CLI arguments, and starts the uvicorn server or fetches USDA labels. The underscore prefix signals it is an internal implementation detail, not part of the public API.

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