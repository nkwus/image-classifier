"""Server-side Pydantic models for the classification API."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class JobStatus(str, Enum):
    """Lifecycle states for a classification job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobSubmission(BaseModel):
    """Returned on ``POST /classify``."""

    model_config = ConfigDict(strict=True)

    job_id: str
    status: JobStatus


class JobResponse(BaseModel):
    """Returned on ``GET /jobs/{job_id}``."""

    model_config = ConfigDict(strict=True)

    job_id: str
    status: JobStatus
    label: str | None = None
    confidence: float | None = None
    progress_pct: int = Field(default=0, ge=0, le=100)
    eta_seconds: float | None = None
    created_at: str
