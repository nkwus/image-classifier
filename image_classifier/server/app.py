"""FastAPI application — instance + lifespan only.

No route handlers or business logic belongs in this file.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import redis as redis_mod
from fastapi import FastAPI

from image_classifier.server.routes import router


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Startup / shutdown lifecycle hook."""
    # --- startup -----------------------------------------------------------
    redis_url = os.environ.get(
        "REDIS_URL",
        "redis://redis-15281.c275.us-east-1-4.ec2.cloud.redislabs.com:15281/0",
    )
    redis_username = os.environ.get("REDIS_USERNAME")
    redis_password = os.environ.get("REDIS_PASSWORD")

    app.state.redis_sync = redis_mod.Redis.from_url(
        redis_url,
        username=redis_username,
        password=redis_password,
        decode_responses=False,
    )
    app.state.executor = ThreadPoolExecutor(max_workers=1)

    # Pre-load the classification model so the first request doesn't pay the
    # cold-start cost.
    from image_classifier.classify import _get_classifier

    _get_classifier()

    yield

    # --- shutdown ----------------------------------------------------------
    app.state.executor.shutdown(wait=False)
    app.state.redis_sync.close()


app = FastAPI(
    title="Image Classifier API",
    version="0.1.0",
    lifespan=_lifespan,
)
app.include_router(router)
