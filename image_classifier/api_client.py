"""HTTP client for the classification API."""

from __future__ import annotations

import time
from types import TracebackType

import httpx

from image_classifier.client_models import JobResponse, JobStatus, JobSubmission


class ClassifierClient:
    """Synchronous client for the image classification REST API.

    Use as a context manager for automatic resource cleanup::

        with ClassifierClient("http://127.0.0.1:8000") as client:
            sub = client.submit(image_bytes)
            ...
    """

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self._base_url, timeout=30.0)

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> ClassifierClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        self._http.close()

    # -- API methods --------------------------------------------------------

    def submit(self, image_bytes: bytes) -> JobSubmission:
        """Upload an image for classification. Returns a ``JobSubmission``."""
        resp = self._http.post(
            "/classify",
            files={"file": ("image.png", image_bytes, "image/png")},
        )
        resp.raise_for_status()
        return JobSubmission.model_validate_json(resp.text)

    def poll(self, job_id: str) -> JobResponse:
        """Fetch the current state of a classification job."""
        resp = self._http.get(f"/jobs/{job_id}")
        resp.raise_for_status()
        return JobResponse.model_validate_json(resp.text)

    def get_image(self, job_id: str) -> bytes | None:
        """Download the annotated result image, or ``None`` if not ready."""
        resp = self._http.get(f"/jobs/{job_id}/image")
        if resp.status_code == 200:
            return resp.content
        return None

    def wait_for_result(
        self,
        job_id: str,
        *,
        timeout: float = 120.0,
        interval: float = 1.0,
    ) -> tuple[bytes, JobResponse] | None:
        """Poll until the job completes or *timeout* seconds elapse.

        Returns ``(annotated_png_bytes, job_response)`` on success, or
        ``None`` on timeout / failure.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.poll(job_id)
            if job.status == JobStatus.COMPLETED:
                image_data = self.get_image(job_id)
                if image_data is not None:
                    return image_data, job
                return None
            if job.status == JobStatus.FAILED:
                return None
            time.sleep(interval)
        return None
