# Plan: Split Image Classifier into Frontend/Backend

## TL;DR

Split the monolithic OpenCV GUI + SigLIP classifier into a **FastAPI REST backend** and a **thin camera frontend**. The backend accepts image uploads as classification jobs, processes them asynchronously via a thread pool, and stores results in Redis. The frontend submits images over HTTP and polls for annotated results. This enables future web frontend replacement.

## Architecture

```
[Camera Frontend]  --POST /classify-->  [FastAPI Server]
       |                                      |
       |           <--GET /jobs/{id}--        |
       |           <--GET /jobs/{id}/image--  |
       |                                      |
       |                              [Thread Pool Executor]
       |                                      |
       |                              [SigLIP Model + Annotate]
       |                                      |
       |                              [Redis: job state + images]
```

## Phase 1: Dependencies & Project Setup

1. Add to `pyproject.toml` main dependencies:
   - `fastapi>=0.115`
   - `uvicorn[standard]>=0.30`
   - `python-multipart>=0.0.9` (file uploads)
   - `pydantic>=2.0` (explicit — used by client_models.py outside of FastAPI)
   - `redis>=5.0` (sync + async Redis clients)
   - `httpx>=0.27` (HTTP client for the frontend)
2. Add to `pyproject.toml` test/dev dependencies:
   - `fakeredis>=2.21` (in-memory Redis for tests)

## Phase 2: Backend — FastAPI Server

3. Create `image_classifier/server/__init__.py` (empty).

4. Create `image_classifier/server/app.py`:
   - FastAPI app instance + lifespan handler only
   - Lifespan: on startup pre-load the SigLIP model via `classify._get_classifier()`, create sync `redis.Redis` connection pool (used by both routes via `to_thread` and tasks directly), create a `ThreadPoolExecutor(max_workers=1)` (model is not thread-safe, single worker serializes inference)
   - Store Redis client and executor on `app.state` for access by routes
   - On shutdown: close Redis pool, shutdown executor
   - Include router from `routes.py`
   - **No route handlers or business logic in this file**

4b. Create `image_classifier/server/routes.py`:
   - Thin HTTP layer — request validation, response formatting, status codes
   - `POST /classify`: Accept `UploadFile`, **validate the upload is a decodable image** (attempt `cv2.imdecode` and return 422 if it fails), generate UUID job_id, store job in Redis as pending, schedule background task via executor, return 202 + `JobSubmission`
   - `GET /jobs/{job_id}`: Lookup job in Redis, return `JobResponse` including `status`, `progress_pct` (0–100), and `eta_seconds` (200 if completed/failed, 202 if pending/running, 404 if not found)
   - `GET /jobs/{job_id}/image`: Return `StreamingResponse` with `media_type="image/png"` if completed, 404/202 otherwise
   - `DELETE /jobs/{job_id}`: Remove from Redis, return 204
   - Delegates all domain work to `tasks.py` — routes contain no classification or image processing logic

4c. Create `image_classifier/server/tasks.py`:
   - Background classification orchestration (pure domain logic, no HTTP concerns)
   - **Synchronous function** — called via `loop.run_in_executor(executor, run_classification, ...)` from routes.py
   - `run_classification(job_id: str, image_bytes: bytes, redis_sync: redis.Redis)` — the single entry point:
     - Takes a **sync** `redis.Redis` client (not async — this runs in a thread pool, not the event loop)
     - Decode image bytes → numpy array via cv2.imdecode
     - Update job: status=running, progress_pct=10, eta_seconds from rolling average
     - Call `classify_food(frame, candidate_labels)` (blocking)
     - Update job: progress_pct=80
     - Call `annotate_image(frame, label, confidence)`
     - Encode annotated frame to PNG bytes via cv2.imencode
     - Store result image + update job: status=completed, progress_pct=100, eta_seconds=0
     - On exception: update job status=failed
     - Track inference duration for ETA (rolling average in Redis key `stats:avg_duration`)
   - Testable independently from HTTP layer (inject mock Redis client)

5. Create `image_classifier/server/models.py` — **server-side Pydantic models** (the server owns its own schema):
   - `JobStatus(str, Enum)`: `pending`, `running`, `completed`, `failed`
   - `JobSubmission(BaseModel)`: `job_id: str`, `status: JobStatus` — returned on POST /classify
   - `JobResponse(BaseModel)`: `job_id: str`, `status: JobStatus`, `label: str | None`, `confidence: float | None`, `progress_pct: int` (Field ge=0, le=100), `eta_seconds: float | None`, `created_at: str` — returned on GET /jobs/{id}
   - All models use `model_config = ConfigDict(strict=True)` for strict type coercion
   - These models are **internal to the server** and NOT imported by the client

5b. Create `image_classifier/client_models.py` — **client-side Pydantic models** (independently defined, no server imports):
   - `JobStatus(str, Enum)`: mirrors server values (`pending`, `running`, `completed`, `failed`)
   - `JobSubmission(BaseModel)`: `job_id: str`, `status: JobStatus`
   - `JobResponse(BaseModel)`: `job_id: str`, `status: JobStatus`, `label: str | None`, `confidence: float | None`, `progress_pct: int` (Field ge=0, le=100), `eta_seconds: float | None`, `created_at: str`
   - All models use `model_config = ConfigDict(strict=True)` for strict type coercion
   - The only contract between server and client is the **REST API JSON schema** — no shared code

6. Create `image_classifier/server/jobs.py` — Redis-backed job store:
   - **All functions are synchronous** (accept `redis.Redis`, not `redis.asyncio.Redis`)
   - Used directly by `tasks.py` (sync thread pool) and wrapped via `asyncio.to_thread()` by routes
   - `create_job(redis, job_id)` → store hash `job:{job_id}` with status=pending, progress_pct=0, eta_seconds=None, created_at, TTL=1hr
   - `update_job(redis, job_id, status, label, confidence, progress_pct, eta_seconds)` → update hash fields
   - `store_result_image(redis, job_id, png_bytes)` → store as `job:{job_id}:image`, TTL=1hr
   - `get_job(redis, job_id)` → return job metadata dict or None
   - `get_result_image(redis, job_id)` → return bytes or None
   - `delete_job(redis, job_id)` → delete both keys

7. Routes in `routes.py` call `jobs` functions via `asyncio.to_thread()` and dispatch classification via `run_in_executor`:
   ```python
   # POST /classify — route creates job, then dispatches task
   await asyncio.to_thread(jobs.create_job, app.state.redis_sync, job_id)
   loop.run_in_executor(app.state.executor, run_classification, job_id, image_bytes, app.state.redis_sync)

   # GET /jobs/{id} — route reads job via sync jobs.py wrapped in to_thread
   job = await asyncio.to_thread(jobs.get_job, app.state.redis_sync, job_id)
   ```
   This keeps `jobs.py` purely synchronous (single implementation) while routes remain non-blocking.
   Only one Redis client needed: sync `redis.Redis` stored on `app.state.redis_sync`.

## Phase 3: CLI Integration

9. Update `_cli.py`:
   - Add `serve` subcommand that starts uvicorn with the FastAPI app
   - Accept `--host` (default `127.0.0.1`) and `--port` (default `8000`) options
   - Keep existing `--fetch-labels` and default camera mode

## Phase 4: API Client

10. Create `image_classifier/api_client.py`:
    - Class `ClassifierClient` wrapping httpx
    - `__init__(self, base_url: str)` — store base URL, create httpx.Client
    - `submit(self, image_bytes: bytes) -> JobSubmission` — POST to /classify, validate response into Pydantic `JobSubmission` model
    - `poll(self, job_id: str) -> JobResponse` — GET /jobs/{job_id}, validate response into Pydantic `JobResponse` model
    - `get_image(self, job_id: str) -> bytes | None` — GET /jobs/{job_id}/image
    - `wait_for_result(self, job_id: str, timeout: float, interval: float) -> tuple[bytes, JobResponse] | None` — convenience: submit+poll loop, returns annotated image bytes + validated `JobResponse`
    - All HTTP responses deserialized and validated through the client's own Pydantic models from `image_classifier.client_models` — no server imports
    - Context manager support for proper cleanup

## Phase 5: Frontend Adaptation

11. Modify `camera.py`:
    - `_classify_worker` now uses `ClassifierClient` instead of calling `classify_food()` + `annotate_image()` directly
    - Submit frame as PNG bytes via client.submit()
    - Poll in a loop (or use wait_for_result)
    - Decode returned annotated image, save via `save_frame()`
    - Accept server URL as parameter to `capture_from_webcam()` (added param with default from env var `CLASSIFIER_API_URL` or fallback default `http://127.0.0.1:8000`)

12. `__init__.py` — no changes needed. `ClassifierClient` is imported directly from `image_classifier.api_client` by consumers. The classification domain exports (`classify_food`, `annotate_image`, `save_frame`, `capture_from_webcam`) remain as-is.

## Phase 6: Tests

13. Server route tests (`tests/test_server_routes.py`):
    - Use FastAPI `TestClient` (sync) or `httpx.AsyncClient` with `ASGITransport`
    - Mock Redis with fakeredis, mock `tasks.run_classification` to avoid model loading
    - Test: submit image → 202, submit non-image file → 422, poll pending → 202, poll completed → 200 with metadata, get image → PNG bytes, not found → 404, delete → 204

13b. Server task tests (`tests/test_server_tasks.py`):
    - Test `run_classification()` in isolation — mock `classify_food`, `annotate_image`, and Redis
    - Verify progress updates (pending → running → completed), image encoding, failure handling
    - Verify ETA rolling average computation

13c. Job store tests (`tests/test_server_jobs.py`):
    - Use fakeredis
    - Test create_job, update_job, get_job, store/get result_image, delete_job
    - Test TTL is set correctly
    - Test get_job returns None for missing keys

13d. Model validation tests (`tests/test_server_models.py` and `tests/test_client_models.py`):
    - Test `progress_pct` rejects values <0 and >100
    - Test `JobStatus` enum rejects invalid values
    - Test strict mode rejects wrong types (e.g. string for int)
    - Test round-trip: dict → model → dict

14. API client tests (`tests/test_api_client.py`):
    - Mock httpx responses
    - Test submit, poll, get_image, wait_for_result timeout

15. Update `tests/test_camera.py`:
    - Mock `ClassifierClient` instead of `classify_food`

16. Update `tests/test_cli.py`:
    - Test `serve` subcommand triggers uvicorn

## Relevant Files

### Existing (to modify)
- `pyproject.toml` — add dependencies and optional `serve` entry point
- `image_classifier/_cli.py` — add `serve` subcommand with `--host`/`--port`
- `image_classifier/camera.py` — replace local classification with API client calls in `_classify_worker()`
- `tests/test_camera.py` — mock API client instead of classify_food
- `tests/test_cli.py` — test serve subcommand

### New (to create)
- `image_classifier/server/__init__.py`
- `image_classifier/server/app.py` — FastAPI instance + lifespan only
- `image_classifier/server/routes.py` — thin HTTP route handlers
- `image_classifier/server/tasks.py` — background classification orchestration
- `image_classifier/server/models.py` — server-side Pydantic models (JobStatus, JobResponse, JobSubmission)
- `image_classifier/server/jobs.py` — Redis-backed job CRUD
- `image_classifier/client_models.py` — client-side Pydantic models (independently defined, mirrors API contract)
- `image_classifier/api_client.py` — httpx-based ClassifierClient
- `tests/test_server_routes.py` — HTTP endpoint tests
- `tests/test_server_tasks.py` — background task tests
- `tests/test_server_jobs.py` — Redis job store tests
- `tests/test_server_models.py` — server Pydantic model validation tests
- `tests/test_client_models.py` — client Pydantic model validation tests
- `tests/test_api_client.py` — client HTTP tests
- `tests/conftest.py` — shared fixtures (fakeredis instance reused by test_server_routes, test_server_tasks, test_server_jobs)

## Verification

1. `pytest` — all existing tests still pass (camera tests updated to mock API client)
2. Ensure Redis Cloud is reachable (`REDIS_URL` and `REDIS_PASSWORD` set in `.env`), run `image-classifier serve`, POST an image with curl:
   ```
   curl -X POST http://127.0.0.1:8000/classify -F "file=@captures/capture_20260412_123456.png"
   ```
   Verify 202 response with job_id.
3. Poll with `curl http://127.0.0.1:8000/jobs/{job_id}` — verify status transitions pending→running→completed.
4. Fetch annotated image: `curl http://127.0.0.1:8000/jobs/{job_id}/image -o result.png` — verify annotated PNG.
5. Run full end-to-end: `image-classifier serve` in one terminal, `image-classifier` in another — verify camera captures classify via API.
6. `mypy --strict` or Pylance strict mode — no type errors in new code.

## Decisions

- **Async pattern:** Job queue (submit POST → poll GET)
- **Storage:** Redis for job state and result images, with 1-hour TTL
- **Thread pool:** Single-worker executor since SigLIP model is not thread-safe; serializes inference
- **Scope included:** REST API, Redis job store, API client, camera adaptation, CLI serve command, tests
- **Scope excluded:** Web frontend (future), authentication/authorization, rate limiting, Docker/deployment config, WebSocket notifications
- **Backward compatibility:** existing `classify_food()` and `annotate_image()` remain importable and functional for direct use

## Further Considerations

1. **Redis connection config:** `REDIS_URL` env var (default `redis://redis-15281.c275.us-east-1-4.ec2.cloud.redislabs.com:15281/0`) loaded from `.env` alongside existing `FDA_API_KEY`. Redis Cloud password stored in `REDIS_PASSWORD` env var.
2. **Concurrent classification:** Current plan uses a single-worker thread pool. If GPU supports it or multiple GPUs exist, this could be expanded. Start with 1 for safety.
3. **WebSocket push:** Instead of polling, a WebSocket endpoint could push results when ready. Recommend deferring this to the web frontend phase since the camera client is fine with polling.
