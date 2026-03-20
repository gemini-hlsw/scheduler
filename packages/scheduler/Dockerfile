FROM python:3.11-slim

# Default app version is development
ARG APP_VERSION="development"

ENV APP_VERSION=$APP_VERSION

WORKDIR /home

# Manage dependencies
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache

COPY ./scheduler /home/scheduler
COPY ./definitions.py /home/definitions.py

RUN apt-get update && apt-get install -y bzip2 && rm -rf /var/lib/apt/lists/*
RUN tar -xjf /home/scheduler/services/horizons/data/ephemerides.tar.bz2 -C /home/scheduler/services/horizons/data

ENV PYTHONPATH="${PYTHONPATH}:/scheduler"
ENTRYPOINT ["uv", "run", "python", "/home/scheduler/main.py" ]
