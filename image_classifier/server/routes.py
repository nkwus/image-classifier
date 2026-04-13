"""Thin HTTP route handlers — request validation, response formatting."""

from __future__ import annotations

import asyncio
import io
import json
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, Request, Response, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from image_classifier.server import jobs
from image_classifier.server.models import JobResponse, JobStatus, JobSubmission
from image_classifier.server.tasks import run_classification

router = APIRouter()


@router.post(
    "/classify",
    response_model=JobSubmission,
    status_code=202,
)
async def submit_classification(file: UploadFile, request: Request) -> Response:
    """Accept an image upload and queue it for classification."""
    image_bytes = await file.read()

    # Validate the upload is a decodable image.
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    test_decode = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if test_decode is None:
        return JSONResponse(
            status_code=422,
            content={"detail": "Uploaded file is not a valid image."},
        )

    job_id = uuid.uuid4().hex
    redis_sync = request.app.state.redis_sync

    await asyncio.to_thread(jobs.create_job, redis_sync, job_id)

    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        request.app.state.executor,
        run_classification,
        job_id,
        image_bytes,
        redis_sync,
    )

    body = JobSubmission(job_id=job_id, status=JobStatus.PENDING)
    return JSONResponse(status_code=202, content=body.model_dump())


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str, request: Request) -> Response:
    """Return the current state of a classification job."""
    redis_sync = request.app.state.redis_sync
    data = await asyncio.to_thread(jobs.get_job, redis_sync, job_id)
    if data is None:
        return JSONResponse(status_code=404, content={"detail": "Job not found."})

    job = JobResponse.model_validate_json(json.dumps(data))
    status_code = 200 if job.status in (JobStatus.COMPLETED, JobStatus.FAILED) else 202
    return JSONResponse(status_code=status_code, content=job.model_dump())


@router.get("/jobs/{job_id}/image")
async def get_job_image(job_id: str, request: Request) -> Response:
    """Return the annotated result image for a completed job."""
    redis_sync = request.app.state.redis_sync
    data = await asyncio.to_thread(jobs.get_job, redis_sync, job_id)
    if data is None:
        return JSONResponse(status_code=404, content={"detail": "Job not found."})

    if data["status"] != "completed":
        return JSONResponse(
            status_code=202,
            content={"detail": "Job not yet completed.", "status": data["status"]},
        )

    image_bytes = await asyncio.to_thread(jobs.get_result_image, redis_sync, job_id)
    if image_bytes is None:
        return JSONResponse(status_code=404, content={"detail": "Result image not found."})

    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")


@router.delete("/jobs/{job_id}", status_code=204)
async def delete_job(job_id: str, request: Request) -> Response:
    """Remove a job and its result image from Redis."""
    redis_sync = request.app.state.redis_sync
    await asyncio.to_thread(jobs.delete_job, redis_sync, job_id)
    return Response(status_code=204)
