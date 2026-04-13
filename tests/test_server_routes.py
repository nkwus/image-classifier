"""Tests for the FastAPI server routes."""

from __future__ import annotations

import time
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import cv2
import fakeredis
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from image_classifier.server.jobs import create_job, store_result_image, update_job
from image_classifier.server.routes import router


def _make_png_bytes() -> bytes:
    """Return valid PNG bytes for a small test image."""
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _setup_app() -> TestClient:
    """Create a TestClient with a fakeredis backend and mocked executor.

    Uses a fresh FastAPI instance with only the router — no lifespan hook —
    so the real ML model is never loaded during tests.
    """
    fake_r = fakeredis.FakeRedis()
    test_app = FastAPI()
    test_app.include_router(router)
    test_app.state.redis_sync = fake_r

    # Use a mock executor that runs the task synchronously for tests.
    mock_executor = MagicMock()

    def run_sync(fn: object, *args: object) -> Future[None]:
        assert callable(fn)
        fn(*args)
        fut: Future[None] = Future()
        fut.set_result(None)
        return fut

    mock_executor.submit = run_sync
    test_app.state.executor = mock_executor

    return TestClient(test_app, raise_server_exceptions=False)


class TestSubmitClassification:
    def test_returns_202_with_job_id(self) -> None:
        client = _setup_app()
        png = _make_png_bytes()
        with patch(
            "image_classifier.server.routes.run_classification",
        ):
            resp = client.post("/classify", files={"file": ("img.png", png, "image/png")})
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "pending"

    def test_rejects_non_image_with_422(self) -> None:
        client = _setup_app()
        resp = client.post(
            "/classify",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code == 422


class TestGetJobStatus:
    def test_pending_returns_202(self) -> None:
        client = _setup_app()
        r = client.app.state.redis_sync
        create_job(r, "j1")
        resp = client.get("/jobs/j1")
        assert resp.status_code == 202
        assert resp.json()["status"] == "pending"

    def test_completed_returns_200(self) -> None:
        client = _setup_app()
        r = client.app.state.redis_sync
        create_job(r, "j1")
        update_job(
            r, "j1",
            status="completed",
            label="Pizza",
            confidence=0.95,
            progress_pct=100,
            eta_seconds=0.0,
        )
        resp = client.get("/jobs/j1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["label"] == "Pizza"
        assert data["confidence"] == 0.95
        assert data["progress_pct"] == 100

    def test_not_found_returns_404(self) -> None:
        client = _setup_app()
        resp = client.get("/jobs/nonexistent")
        assert resp.status_code == 404


class TestGetJobImage:
    def test_returns_png_when_completed(self) -> None:
        client = _setup_app()
        r = client.app.state.redis_sync
        create_job(r, "j1")
        update_job(r, "j1", status="completed")
        png = _make_png_bytes()
        store_result_image(r, "j1", png)

        resp = client.get("/jobs/j1/image")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert resp.content == png

    def test_returns_202_when_not_completed(self) -> None:
        client = _setup_app()
        r = client.app.state.redis_sync
        create_job(r, "j1")
        resp = client.get("/jobs/j1/image")
        assert resp.status_code == 202

    def test_returns_404_for_missing_job(self) -> None:
        client = _setup_app()
        resp = client.get("/jobs/nonexistent/image")
        assert resp.status_code == 404


class TestDeleteJob:
    def test_returns_204(self) -> None:
        client = _setup_app()
        r = client.app.state.redis_sync
        create_job(r, "j1")
        resp = client.delete("/jobs/j1")
        assert resp.status_code == 204

    def test_delete_removes_data(self) -> None:
        client = _setup_app()
        r = client.app.state.redis_sync
        create_job(r, "j1")
        store_result_image(r, "j1", b"png")
        client.delete("/jobs/j1")
        resp = client.get("/jobs/j1")
        assert resp.status_code == 404
