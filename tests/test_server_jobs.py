"""Tests for the Redis-backed job store."""

from __future__ import annotations

import redis as redis_mod

from image_classifier.server.jobs import (
    create_job,
    delete_all_jobs,
    delete_job,
    get_job,
    get_result_image,
    list_jobs,
    store_result_image,
    update_job,
)


class TestCreateJob:
    def test_creates_pending_job(self, fake_redis: redis_mod.Redis[bytes]) -> None:
        create_job(fake_redis, "job1")
        data = get_job(fake_redis, "job1")
        assert data is not None
        assert data["job_id"] == "job1"
        assert data["status"] == "pending"
        assert data["progress_pct"] == 0
        assert data["label"] is None
        assert data["confidence"] is None
        assert data["created_at"] != ""

    def test_sets_ttl(self, fake_redis: redis_mod.Redis[bytes]) -> None:
        create_job(fake_redis, "job1")
        ttl = fake_redis.ttl("job:job1")
        assert ttl > 0
        assert ttl <= 3600


class TestUpdateJob:
    def test_updates_status(self, fake_redis: redis_mod.Redis[bytes]) -> None:
        create_job(fake_redis, "job1")
        update_job(fake_redis, "job1", status="running", progress_pct=10)
        data = get_job(fake_redis, "job1")
        assert data is not None
        assert data["status"] == "running"
        assert data["progress_pct"] == 10

    def test_updates_result_fields(self, fake_redis: redis_mod.Redis[bytes]) -> None:
        create_job(fake_redis, "job1")
        update_job(
            fake_redis,
            "job1",
            status="completed",
            label="Pizza",
            confidence=0.95,
            progress_pct=100,
            eta_seconds=0.0,
        )
        data = get_job(fake_redis, "job1")
        assert data is not None
        assert data["label"] == "Pizza"
        assert data["confidence"] == 0.95
        assert data["progress_pct"] == 100
        assert data["eta_seconds"] == 0.0


class TestGetJob:
    def test_returns_none_for_missing(self, fake_redis: redis_mod.Redis[bytes]) -> None:
        assert get_job(fake_redis, "nonexistent") is None


class TestResultImage:
    def test_store_and_retrieve(self, fake_redis: redis_mod.Redis[bytes]) -> None:
        png = b"\x89PNG_fake_image_data"
        store_result_image(fake_redis, "job1", png)
        assert get_result_image(fake_redis, "job1") == png

    def test_returns_none_when_missing(self, fake_redis: redis_mod.Redis[bytes]) -> None:
        assert get_result_image(fake_redis, "nonexistent") is None

    def test_image_has_ttl(self, fake_redis: redis_mod.Redis[bytes]) -> None:
        store_result_image(fake_redis, "job1", b"data")
        ttl = fake_redis.ttl("job:job1:image")
        assert ttl > 0
        assert ttl <= 3600


class TestDeleteJob:
    def test_removes_job_and_image(self, fake_redis: redis_mod.Redis[bytes]) -> None:
        create_job(fake_redis, "job1")
        store_result_image(fake_redis, "job1", b"img")
        delete_job(fake_redis, "job1")
        assert get_job(fake_redis, "job1") is None
        assert get_result_image(fake_redis, "job1") is None


class TestListJobs:
    def test_empty_when_no_jobs(self, fake_redis: redis_mod.Redis[bytes]) -> None:
        assert list_jobs(fake_redis) == []

    def test_returns_all_active_jobs(self, fake_redis: redis_mod.Redis[bytes]) -> None:
        create_job(fake_redis, "job1")
        create_job(fake_redis, "job2")
        update_job(fake_redis, "job1", status="running")
        result = list_jobs(fake_redis)
        ids = {j["job_id"] for j in result}
        assert ids == {"job1", "job2"}

    def test_excludes_image_keys(self, fake_redis: redis_mod.Redis[bytes]) -> None:
        create_job(fake_redis, "job1")
        store_result_image(fake_redis, "job1", b"img")
        result = list_jobs(fake_redis)
        assert len(result) == 1
        assert result[0]["job_id"] == "job1"


class TestDeleteAllJobs:
    def test_deletes_all_jobs_and_images(self, fake_redis: redis_mod.Redis[bytes]) -> None:
        create_job(fake_redis, "job1")
        create_job(fake_redis, "job2")
        store_result_image(fake_redis, "job1", b"img")
        count = delete_all_jobs(fake_redis)
        assert count == 3  # 2 job hashes + 1 image key
        assert get_job(fake_redis, "job1") is None
        assert get_job(fake_redis, "job2") is None
        assert get_result_image(fake_redis, "job1") is None

    def test_returns_zero_when_empty(self, fake_redis: redis_mod.Redis[bytes]) -> None:
        assert delete_all_jobs(fake_redis) == 0
