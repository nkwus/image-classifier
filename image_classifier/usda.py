"""Fetch food labels from the USDA FoodData Central API.

Uses the ``/v1/foods/list`` endpoint to page through Foundation and SR Legacy
foods, collecting their descriptions for use as zero-shot classifier labels.
Labels are cached locally so the API is only hit when explicitly requested.

See https://fdc.nal.usda.gov/api-guide for API documentation.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API_BASE_URL = "https://api.nal.usda.gov/fdc/v1"
DEFAULT_DATA_TYPES: list[str] = ["Foundation", "SR Legacy"]
_PAGE_SIZE = 200  # maximum allowed by the API
_MAX_RETRIES = 3
_RETRY_WAIT_SECONDS = 60  # default wait when Retry-After header is absent
_CACHE_DIR = Path.home() / ".cache" / "image-classifier"
CACHE_PATH = _CACHE_DIR / "usda_labels.json"


def _fetch_page(
    api_key: str,
    data_types: list[str],
    page_number: int,
    page_size: int = _PAGE_SIZE,
) -> list[dict[str, Any]]:
    """Fetch a single page from the ``/foods/list`` endpoint.

    The API key is sent via the ``X-Api-Key`` header rather than as a query
    parameter so it does not leak into server logs or browser history.

    Retries up to :data:`_MAX_RETRIES` times on HTTP 429 (rate-limited),
    respecting the ``Retry-After`` header when present.
    """
    body = json.dumps(
        {
            "dataType": data_types,
            "pageSize": page_size,
            "pageNumber": page_number,
            "sortBy": "lowercaseDescription.keyword",
            "sortOrder": "asc",
        }
    ).encode()

    for attempt in range(_MAX_RETRIES + 1):
        req = urllib.request.Request(
            f"{API_BASE_URL}/foods/list",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Api-Key": api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:  # noqa: S310
                result: list[dict[str, Any]] = json.loads(resp.read())
            return result
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == _MAX_RETRIES:
                raise
            retry_after = exc.headers.get("Retry-After")
            wait = int(retry_after) if retry_after else _RETRY_WAIT_SECONDS
            print(
                f"  Rate-limited (429). Waiting {wait}s before retry "
                f"({attempt + 1}/{_MAX_RETRIES})…",
                file=sys.stderr,
            )
            time.sleep(wait)
    # Unreachable — the loop always returns or raises.
    raise RuntimeError("unreachable")  # pragma: no cover


def fetch_labels(
    api_key: str,
    data_types: list[str] | None = None,
) -> list[str]:
    """Page through ``/foods/list`` and return sorted, deduplicated labels.

    *data_types* defaults to ``["Foundation", "SR Legacy"]``.
    """
    if data_types is None:
        data_types = DEFAULT_DATA_TYPES
    labels: set[str] = set()
    page = 1
    while True:
        items = _fetch_page(api_key, data_types, page)
        if not items:
            break
        for item in items:
            desc = item.get("description", "").strip()
            if desc:
                labels.add(desc.lower())
        print(
            f"  Fetched page {page} ({len(items)} items, "
            f"{len(labels)} unique so far)…",
            file=sys.stderr,
        )
        page += 1
    return sorted(labels)


def cache_labels(labels: list[str], path: Path = CACHE_PATH) -> None:
    """Write *labels* to a JSON cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, indent=2), encoding="utf-8")


def load_cached_labels(path: Path = CACHE_PATH) -> list[str] | None:
    """Load labels from *path*, or return ``None`` if the cache is missing."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(data, list) and all(isinstance(item, str) for item in data):
        return data
    return None


def fetch_and_cache_labels(
    api_key: str | None = None,
    data_types: list[str] | None = None,
    cache_path: Path = CACHE_PATH,
) -> list[str]:
    """Fetch labels from USDA FoodData Central and write them to *cache_path*.

    If *api_key* is ``None``, reads ``FDA_API_KEY`` from the environment.
    Returns the fetched label list (empty if the key is missing).
    """
    if api_key is None:
        api_key = os.environ.get("FDA_API_KEY", "")
    if not api_key:
        print(
            "Error: FDA_API_KEY is not set. "
            "Add it to .env or export it in your shell.",
            file=sys.stderr,
        )
        return []
    print("Fetching food labels from USDA FoodData Central…", file=sys.stderr)
    labels = fetch_labels(api_key, data_types)
    cache_labels(labels, cache_path)
    print(f"Cached {len(labels)} labels → {cache_path}", file=sys.stderr)
    return labels
