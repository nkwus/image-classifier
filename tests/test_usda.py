from __future__ import annotations

import json
import urllib.error
from email.message import Message
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import image_classifier.usda as usda_mod
from image_classifier.usda import (
    API_BASE_URL,
    CACHE_PATH,
    DEFAULT_DATA_TYPES,
    _MAX_RETRIES,
    cache_labels,
    fetch_and_cache_labels,
    fetch_labels,
    load_cached_labels,
)


def _make_items(descriptions: list[str]) -> list[dict[str, Any]]:
    """Build fake AbridgedFoodItem dicts."""
    return [
        {"fdcId": i, "dataType": "Foundation", "description": d}
        for i, d in enumerate(descriptions, start=1)
    ]


class TestFetchPage:
    """Tests for the _fetch_page helper."""

    def test_sends_correct_request(self) -> None:
        fake_items = _make_items(["Apple"])
        fake_resp = MagicMock()
        fake_resp.read.return_value = json.dumps(fake_items).encode()
        fake_resp.__enter__ = MagicMock(return_value=fake_resp)
        fake_resp.__exit__ = MagicMock(return_value=False)

        with patch("image_classifier.usda.urllib.request.urlopen", return_value=fake_resp) as mock_open:
            result = usda_mod._fetch_page("my-key", ["Foundation"], page_number=3, page_size=50)

        assert result == fake_items
        call_args = mock_open.call_args[0][0]
        assert call_args.full_url == f"{API_BASE_URL}/foods/list"
        assert call_args.get_header("X-api-key") == "my-key"
        assert call_args.get_header("Content-type") == "application/json"
        body = json.loads(call_args.data)
        assert body["dataType"] == ["Foundation"]
        assert body["pageSize"] == 50
        assert body["pageNumber"] == 3

    def test_retries_on_429_with_retry_after_header(self) -> None:
        headers = Message()
        headers["Retry-After"] = "2"
        error_429 = urllib.error.HTTPError(
            url="", code=429, msg="Too Many Requests", hdrs=headers, fp=None
        )
        fake_items = _make_items(["Apple"])
        fake_resp = MagicMock()
        fake_resp.read.return_value = json.dumps(fake_items).encode()
        fake_resp.__enter__ = MagicMock(return_value=fake_resp)
        fake_resp.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "image_classifier.usda.urllib.request.urlopen",
                side_effect=[error_429, fake_resp],
            ),
            patch("image_classifier.usda.time.sleep") as mock_sleep,
        ):
            result = usda_mod._fetch_page("key", ["Foundation"], page_number=1)

        assert result == fake_items
        mock_sleep.assert_called_once_with(2)

    def test_retries_on_429_with_default_wait(self) -> None:
        headers = Message()
        error_429 = urllib.error.HTTPError(
            url="", code=429, msg="Too Many Requests", hdrs=headers, fp=None
        )
        fake_items = _make_items(["Banana"])
        fake_resp = MagicMock()
        fake_resp.read.return_value = json.dumps(fake_items).encode()
        fake_resp.__enter__ = MagicMock(return_value=fake_resp)
        fake_resp.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "image_classifier.usda.urllib.request.urlopen",
                side_effect=[error_429, fake_resp],
            ),
            patch("image_classifier.usda.time.sleep") as mock_sleep,
        ):
            result = usda_mod._fetch_page("key", ["Foundation"], page_number=1)

        assert result == fake_items
        mock_sleep.assert_called_once_with(usda_mod._RETRY_WAIT_SECONDS)

    def test_raises_after_max_retries_exceeded(self) -> None:
        headers = Message()
        error_429 = urllib.error.HTTPError(
            url="", code=429, msg="Too Many Requests", hdrs=headers, fp=None
        )

        with (
            patch(
                "image_classifier.usda.urllib.request.urlopen",
                side_effect=[error_429] * (_MAX_RETRIES + 1),
            ),
            patch("image_classifier.usda.time.sleep"),
            pytest.raises(urllib.error.HTTPError, match="Too Many Requests"),
        ):
            usda_mod._fetch_page("key", ["Foundation"], page_number=1)

    def test_non_429_error_raises_immediately(self) -> None:
        headers = Message()
        error_500 = urllib.error.HTTPError(
            url="", code=500, msg="Internal Server Error", hdrs=headers, fp=None
        )

        with (
            patch(
                "image_classifier.usda.urllib.request.urlopen",
                side_effect=error_500,
            ),
            pytest.raises(urllib.error.HTTPError, match="Internal Server Error"),
        ):
            usda_mod._fetch_page("key", ["Foundation"], page_number=1)


class TestFetchLabels:
    """Tests for the fetch_labels function."""

    def test_paginates_and_deduplicates(self) -> None:
        page1 = _make_items(["Strawberries, raw", "Broccoli, raw"])
        page2 = _make_items(["Strawberries, raw", "Cheese, cheddar"])
        page3: list[dict[str, Any]] = []

        with patch.object(usda_mod, "_fetch_page", side_effect=[page1, page2, page3]):
            labels = fetch_labels("key123")

        assert labels == ["broccoli, raw", "cheese, cheddar", "strawberries, raw"]

    def test_uses_default_data_types(self) -> None:
        with patch.object(usda_mod, "_fetch_page", return_value=[]) as mock_fp:
            fetch_labels("key123")

        _, kwargs = mock_fp.call_args
        # First positional arg after api_key is data_types
        args = mock_fp.call_args[0]
        assert args[1] == DEFAULT_DATA_TYPES

    def test_custom_data_types(self) -> None:
        with patch.object(usda_mod, "_fetch_page", return_value=[]) as mock_fp:
            fetch_labels("key123", data_types=["Branded"])

        args = mock_fp.call_args[0]
        assert args[1] == ["Branded"]

    def test_skips_empty_descriptions(self) -> None:
        items = [
            {"fdcId": 1, "dataType": "Foundation", "description": "Apple"},
            {"fdcId": 2, "dataType": "Foundation", "description": ""},
            {"fdcId": 3, "dataType": "Foundation", "description": "  "},
        ]
        with patch.object(usda_mod, "_fetch_page", side_effect=[items, []]):
            labels = fetch_labels("key123")

        assert labels == ["apple"]


class TestCacheLabels:
    """Tests for cache_labels and load_cached_labels."""

    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.json"
        original = ["apple", "banana", "cherry"]
        cache_labels(original, path)
        loaded = load_cached_labels(path)
        assert loaded == original

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "labels.json"
        cache_labels(["test"], path)
        assert path.is_file()

    def test_load_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        assert load_cached_labels(tmp_path / "nonexistent.json") is None

    def test_load_returns_none_for_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not json", encoding="utf-8")
        assert load_cached_labels(path) is None

    def test_load_returns_none_for_wrong_structure(self, tmp_path: Path) -> None:
        path = tmp_path / "wrong.json"
        path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
        assert load_cached_labels(path) is None

    def test_load_returns_none_for_non_string_items(self, tmp_path: Path) -> None:
        path = tmp_path / "mixed.json"
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        assert load_cached_labels(path) is None


class TestFetchAndCacheLabels:
    """Tests for the top-level fetch_and_cache_labels function."""

    def test_fetches_and_caches(self, tmp_path: Path) -> None:
        cache = tmp_path / "labels.json"
        with patch.object(
            usda_mod, "fetch_labels", return_value=["alpha", "beta"]
        ) as mock_fetch:
            result = fetch_and_cache_labels(
                api_key="test-key", cache_path=cache
            )

        assert result == ["alpha", "beta"]
        assert json.loads(cache.read_text(encoding="utf-8")) == ["alpha", "beta"]
        mock_fetch.assert_called_once_with("test-key", None)

    def test_reads_key_from_env(self, tmp_path: Path) -> None:
        cache = tmp_path / "labels.json"
        with (
            patch.dict("os.environ", {"FDA_API_KEY": "env-key"}),
            patch.object(
                usda_mod, "fetch_labels", return_value=[]
            ) as mock_fetch,
        ):
            fetch_and_cache_labels(cache_path=cache)

        mock_fetch.assert_called_once_with("env-key", None)

    def test_returns_empty_when_no_key(self, tmp_path: Path) -> None:
        cache = tmp_path / "labels.json"
        with patch.dict("os.environ", {}, clear=True):
            result = fetch_and_cache_labels(api_key=None, cache_path=cache)

        assert result == []
        assert not cache.exists()

    def test_passes_custom_data_types(self, tmp_path: Path) -> None:
        cache = tmp_path / "labels.json"
        with patch.object(
            usda_mod, "fetch_labels", return_value=["x"]
        ) as mock_fetch:
            fetch_and_cache_labels(
                api_key="k", data_types=["Branded"], cache_path=cache
            )

        mock_fetch.assert_called_once_with("k", ["Branded"])
