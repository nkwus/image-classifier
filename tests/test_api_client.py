"""Tests for the ClassifierClient."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from image_classifier.api_client import ClassifierClient
from image_classifier.client_models import JobResponse, JobStatus, JobSubmission

_CLIENT_MODULE = "image_classifier.api_client"


class TestSubmit:
    def test_returns_job_submission(self) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 202
        mock_resp.text = json.dumps({"job_id": "abc", "status": "pending"})
        mock_resp.raise_for_status = MagicMock()

        with patch.object(httpx.Client, "post", return_value=mock_resp):
            with ClassifierClient("http://test") as client:
                result = client.submit(b"fake-png")

        assert isinstance(result, JobSubmission)
        assert result.job_id == "abc"
        assert result.status == JobStatus.PENDING


class TestPoll:
    def test_returns_job_response(self) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.text = json.dumps({
            "job_id": "abc",
            "status": "completed",
            "label": "Pizza",
            "confidence": 0.95,
            "progress_pct": 100,
            "eta_seconds": 0.0,
            "created_at": "2026-04-13T00:00:00+00:00",
        })
        mock_resp.raise_for_status = MagicMock()

        with patch.object(httpx.Client, "get", return_value=mock_resp):
            with ClassifierClient("http://test") as client:
                result = client.poll("abc")

        assert isinstance(result, JobResponse)
        assert result.label == "Pizza"
        assert result.status == JobStatus.COMPLETED


class TestGetImage:
    def test_returns_bytes_on_200(self) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.content = b"PNG_DATA"

        with patch.object(httpx.Client, "get", return_value=mock_resp):
            with ClassifierClient("http://test") as client:
                result = client.get_image("abc")

        assert result == b"PNG_DATA"

    def test_returns_none_on_non_200(self) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 202

        with patch.object(httpx.Client, "get", return_value=mock_resp):
            with ClassifierClient("http://test") as client:
                result = client.get_image("abc")

        assert result is None


class TestWaitForResult:
    def test_returns_on_completed(self) -> None:
        poll_resp = MagicMock(spec=httpx.Response)
        poll_resp.status_code = 200
        poll_resp.text = json.dumps({
            "job_id": "abc",
            "status": "completed",
            "label": "Sushi",
            "confidence": 0.8,
            "progress_pct": 100,
            "eta_seconds": 0.0,
            "created_at": "t",
        })
        poll_resp.raise_for_status = MagicMock()

        img_resp = MagicMock(spec=httpx.Response)
        img_resp.status_code = 200
        img_resp.content = b"ANNOTATED"

        with patch.object(
            httpx.Client, "get", side_effect=[poll_resp, img_resp],
        ):
            with ClassifierClient("http://test") as client:
                result = client.wait_for_result("abc", timeout=5.0, interval=0.01)

        assert result is not None
        img_bytes, job = result
        assert img_bytes == b"ANNOTATED"
        assert job.label == "Sushi"

    def test_returns_none_on_failure(self) -> None:
        poll_resp = MagicMock(spec=httpx.Response)
        poll_resp.status_code = 200
        poll_resp.text = json.dumps({
            "job_id": "abc",
            "status": "failed",
            "label": None,
            "confidence": None,
            "progress_pct": 10,
            "eta_seconds": None,
            "created_at": "t",
        })
        poll_resp.raise_for_status = MagicMock()

        with patch.object(httpx.Client, "get", return_value=poll_resp):
            with ClassifierClient("http://test") as client:
                result = client.wait_for_result("abc", timeout=5.0, interval=0.01)

        assert result is None

    def test_returns_none_on_timeout(self) -> None:
        poll_resp = MagicMock(spec=httpx.Response)
        poll_resp.status_code = 200
        poll_resp.text = json.dumps({
            "job_id": "abc",
            "status": "running",
            "label": None,
            "confidence": None,
            "progress_pct": 50,
            "eta_seconds": 5.0,
            "created_at": "t",
        })
        poll_resp.raise_for_status = MagicMock()

        with patch.object(httpx.Client, "get", return_value=poll_resp):
            with ClassifierClient("http://test") as client:
                result = client.wait_for_result("abc", timeout=0.05, interval=0.01)

        assert result is None


class TestContextManager:
    def test_close_is_called(self) -> None:
        with patch.object(httpx.Client, "close") as mock_close:
            with ClassifierClient("http://test"):
                pass
        mock_close.assert_called_once()
