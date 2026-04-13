"""Tests for server-side Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from image_classifier.server.models import JobResponse, JobStatus, JobSubmission


class TestJobStatus:
    def test_valid_values(self) -> None:
        assert JobStatus("pending") == JobStatus.PENDING
        assert JobStatus("running") == JobStatus.RUNNING
        assert JobStatus("completed") == JobStatus.COMPLETED
        assert JobStatus("failed") == JobStatus.FAILED

    def test_invalid_value_rejected(self) -> None:
        with pytest.raises(ValueError):
            JobStatus("bogus")


class TestJobSubmission:
    def test_valid(self) -> None:
        m = JobSubmission(job_id="abc123", status=JobStatus.PENDING)
        assert m.job_id == "abc123"
        assert m.status == JobStatus.PENDING

    def test_rejects_wrong_type_for_job_id(self) -> None:
        with pytest.raises(ValidationError):
            JobSubmission(job_id=123, status=JobStatus.PENDING)  # type: ignore[arg-type]

    def test_round_trip(self) -> None:
        m = JobSubmission(job_id="x", status=JobStatus.RUNNING)
        data = m.model_dump()
        assert data == {"job_id": "x", "status": "running"}
        restored = JobSubmission.model_validate(data)
        assert restored == m


class TestJobResponse:
    def test_valid_completed(self) -> None:
        m = JobResponse(
            job_id="j1",
            status=JobStatus.COMPLETED,
            label="Pizza",
            confidence=0.95,
            progress_pct=100,
            eta_seconds=0.0,
            created_at="2026-04-13T00:00:00+00:00",
        )
        assert m.label == "Pizza"
        assert m.progress_pct == 100

    def test_defaults(self) -> None:
        m = JobResponse(
            job_id="j1",
            status=JobStatus.PENDING,
            created_at="2026-04-13T00:00:00+00:00",
        )
        assert m.label is None
        assert m.confidence is None
        assert m.progress_pct == 0
        assert m.eta_seconds is None

    def test_progress_pct_rejects_below_zero(self) -> None:
        with pytest.raises(ValidationError):
            JobResponse(
                job_id="j1",
                status=JobStatus.PENDING,
                progress_pct=-1,
                created_at="t",
            )

    def test_progress_pct_rejects_above_100(self) -> None:
        with pytest.raises(ValidationError):
            JobResponse(
                job_id="j1",
                status=JobStatus.PENDING,
                progress_pct=101,
                created_at="t",
            )

    def test_strict_rejects_string_for_progress_pct(self) -> None:
        with pytest.raises(ValidationError):
            JobResponse(
                job_id="j1",
                status=JobStatus.PENDING,
                progress_pct="50",  # type: ignore[arg-type]
                created_at="t",
            )

    def test_round_trip(self) -> None:
        m = JobResponse(
            job_id="j1",
            status=JobStatus.COMPLETED,
            label="Sushi",
            confidence=0.85,
            progress_pct=100,
            eta_seconds=0.0,
            created_at="2026-04-13T00:00:00+00:00",
        )
        data = m.model_dump()
        restored = JobResponse.model_validate(data)
        assert restored == m
