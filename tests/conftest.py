"""Shared test fixtures."""

from __future__ import annotations

import fakeredis
import pytest
import redis as redis_mod


@pytest.fixture()
def fake_redis() -> redis_mod.Redis[bytes]:
    """Return a fakeredis instance for unit tests."""
    return fakeredis.FakeRedis()
