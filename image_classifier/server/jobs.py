"""Redis-backed job store for classification jobs.

All functions are **synchronous** (accept ``redis.Redis``).  Routes call
them via ``asyncio.to_thread()``, and ``tasks.py`` calls them directly
from the thread-pool worker.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import redis as redis_mod

_JOB_TTL_SECONDS = 3600  # 1 hour


def create_job(r: redis_mod.Redis[bytes], job_id: str) -> None:
    """Create a new pending job in Redis."""
    key = f"job:{job_id}"
    mapping: dict[str, str] = {
        "status": "pending",
        "progress_pct": "0",
        "eta_seconds": "",
        "label": "",
        "confidence": "",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    r.hset(key, mapping=mapping)
    r.expire(key, _JOB_TTL_SECONDS)


def update_job(
    r: redis_mod.Redis[bytes],
    job_id: str,
    *,
    status: str | None = None,
    label: str | None = None,
    confidence: float | None = None,
    progress_pct: int | None = None,
    eta_seconds: float | None = None,
) -> None:
    """Update selected fields of an existing job hash."""
    key = f"job:{job_id}"
    updates: dict[str, str] = {}
    if status is not None:
        updates["status"] = status
    if label is not None:
        updates["label"] = label
    if confidence is not None:
        updates["confidence"] = str(confidence)
    if progress_pct is not None:
        updates["progress_pct"] = str(progress_pct)
    if eta_seconds is not None:
        updates["eta_seconds"] = str(eta_seconds)
    if updates:
        r.hset(key, mapping=updates)


def store_result_image(
    r: redis_mod.Redis[bytes],
    job_id: str,
    png_bytes: bytes,
) -> None:
    """Store the annotated result image for a completed job."""
    key = f"job:{job_id}:image"
    r.set(key, png_bytes, ex=_JOB_TTL_SECONDS)


def get_job(r: redis_mod.Redis[bytes], job_id: str) -> dict[str, Any] | None:
    """Return job metadata as a dict, or ``None`` if the job does not exist."""
    key = f"job:{job_id}"
    raw: dict[bytes, bytes] = r.hgetall(key)  # type: ignore[assignment]
    if not raw:
        return None
    decoded: dict[str, str] = {
        k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v
        for k, v in raw.items()
    }
    result: dict[str, Any] = {
        "job_id": job_id,
        "status": decoded.get("status", "pending"),
        "label": decoded.get("label") or None,
        "confidence": float(decoded["confidence"]) if decoded.get("confidence") else None,
        "progress_pct": int(decoded.get("progress_pct", "0")),
        "eta_seconds": float(decoded["eta_seconds"]) if decoded.get("eta_seconds") else None,
        "created_at": decoded.get("created_at", ""),
    }
    return result


def get_result_image(r: redis_mod.Redis[bytes], job_id: str) -> bytes | None:
    """Return the annotated PNG bytes, or ``None`` if not available."""
    key = f"job:{job_id}:image"
    data: bytes | None = r.get(key)  # type: ignore[assignment]
    return data


def delete_job(r: redis_mod.Redis[bytes], job_id: str) -> None:
    """Remove a job and its result image from Redis."""
    r.delete(f"job:{job_id}", f"job:{job_id}:image")


def list_jobs(r: redis_mod.Redis[bytes]) -> list[dict[str, Any]]:
    """Return metadata for all active jobs (those matching ``job:*``)."""
    results: list[dict[str, Any]] = []
    cursor: int = 0
    while True:
        cursor, keys = r.scan(cursor=cursor, match="job:*", count=100)  # type: ignore[assignment]
        for key in keys:
            key_str = key.decode() if isinstance(key, bytes) else key
            # Skip image keys
            if key_str.endswith(":image"):
                continue
            job_id = key_str.removeprefix("job:")
            job = get_job(r, job_id)
            if job is not None:
                results.append(job)
        if cursor == 0:
            break
    return results


def delete_all_jobs(r: redis_mod.Redis[bytes]) -> int:
    """Delete all job keys (hashes and images). Returns the count deleted."""
    count = 0
    cursor: int = 0
    while True:
        cursor, keys = r.scan(cursor=cursor, match="job:*", count=100)  # type: ignore[assignment]
        if keys:
            count += r.delete(*keys)  # type: ignore[assignment]
        if cursor == 0:
            break
    return count
