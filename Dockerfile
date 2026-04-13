FROM python:3.12-slim AS base

# System deps for opencv-python-headless
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first (layer cache)
COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project

# Copy application code
COPY image_classifier/ image_classifier/

# Install the project itself
RUN uv sync --no-dev

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

ENTRYPOINT ["python", "-m", "image_classifier._cli"]
