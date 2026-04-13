"""Client-side Pydantic models — independently defined, no server imports.

The only contract between server and client is the REST API JSON schema.
"""

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
    """Response from ``POST /classify``."""

    model_config = ConfigDict(strict=True)

    job_id: str
    status: JobStatus


class JobResponse(BaseModel):
    """Response from ``GET /jobs/{job_id}``."""

    model_config = ConfigDict(strict=True)

    job_id: str
    status: JobStatus
    label: str | None = None
    confidence: float | None = None
    progress_pct: int = Field(default=0, ge=0, le=100)
    eta_seconds: float | None = None
    created_at: str
