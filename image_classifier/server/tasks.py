"""Background classification task — pure domain logic, no HTTP concerns.

The ``run_classification`` function is **synchronous** and is designed to
be dispatched via ``loop.run_in_executor(executor, run_classification, ...)``.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import redis as redis_mod

from image_classifier.annotate import annotate_image
from image_classifier.classify import classify_food
from image_classifier.server import jobs

_STATS_KEY = "stats:avg_duration"


def _get_eta(r: redis_mod.Redis[bytes]) -> float | None:
    """Return the rolling-average inference duration, or ``None``."""
    raw: bytes | None = r.get(_STATS_KEY)  # type: ignore[assignment]
    if raw is None:
        return None
    return float(raw)


def _record_duration(r: redis_mod.Redis[bytes], duration: float) -> None:
    """Update the rolling average inference duration (simple EMA, α=0.3)."""
    prev = _get_eta(r)
    alpha = 0.3
    if prev is None:
        avg = duration
    else:
        avg = alpha * duration + (1 - alpha) * prev
    r.set(_STATS_KEY, str(avg))


def run_classification(
    job_id: str,
    image_bytes: bytes,
    redis_sync: redis_mod.Redis[bytes],
) -> None:
    """Classify an image and store the annotated result in Redis.

    This is a blocking function — call it from a thread-pool executor.

    Parameters
    ----------
    job_id:
        The UUID of the job to update.
    image_bytes:
        Raw bytes of the uploaded image (PNG/JPEG).
    redis_sync:
        A synchronous ``redis.Redis`` client for progress updates.
    """
    try:
        # --- decode --------------------------------------------------------
        buf = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            jobs.update_job(redis_sync, job_id, status="failed")
            return

        # --- running -------------------------------------------------------
        eta = _get_eta(redis_sync)
        jobs.update_job(
            redis_sync,
            job_id,
            status="running",
            progress_pct=10,
            eta_seconds=eta,
        )

        t0 = time.monotonic()
        label, confidence = classify_food(frame)
        duration = time.monotonic() - t0

        # --- post-inference ------------------------------------------------
        jobs.update_job(redis_sync, job_id, progress_pct=80)
        annotated = annotate_image(frame, label, confidence)
        ok, encoded = cv2.imencode(".png", annotated)
        if not ok:
            jobs.update_job(redis_sync, job_id, status="failed")
            return

        png_bytes: bytes = encoded.tobytes()
        jobs.store_result_image(redis_sync, job_id, png_bytes)
        jobs.update_job(
            redis_sync,
            job_id,
            status="completed",
            label=label,
            confidence=confidence,
            progress_pct=100,
            eta_seconds=0.0,
        )

        _record_duration(redis_sync, duration)

    except Exception:
        jobs.update_job(redis_sync, job_id, status="failed")
        raise
