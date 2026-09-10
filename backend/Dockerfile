FROM python:3.11-slim

# Default app version is development
ARG APP_VERSION="development"
ENV APP_VERSION=$APP_VERSION

# Selects which gpp-client track to install: `gpp-prod` (stable, default) or
# `gpp-dev` (.devN pre-releases). DEV deployments override to `gpp-dev`.
ARG GPP_GROUP=gpp-prod

WORKDIR /home

# Manage dependencies
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
COPY backend/pyproject.toml uv.lock ./
RUN if [ "$GPP_GROUP" = "gpp-dev" ]; then \
        uv sync --frozen --no-cache --group gpp-dev --no-group gpp-prod; \
    elif [ "$GPP_GROUP" = "gpp-prod" ]; then \
        uv sync --frozen --no-cache --group gpp-prod --no-group gpp-dev; \
    else \
        echo "Invalid GPP_GROUP=$GPP_GROUP (expected gpp-prod or gpp-dev)" && exit 1; \
    fi

COPY ./backend/scheduler /home/scheduler
COPY ./backend/definitions.py /home/definitions.py
COPY ./version.py /home/version.py
COPY ./backend/alembic /home/alembic
COPY ./backend/alembic.ini /home/alembic.ini

RUN apt-get update && apt-get install -y bzip2 && rm -rf /var/lib/apt/lists/*
RUN tar -xjf /home/scheduler/services/horizons/data/ephemerides.tar.bz2 -C /home/scheduler/services/horizons/data

ENV PYTHONPATH="${PYTHONPATH}:/scheduler"

CMD ["uv", "run", "python", "/home/scheduler/main.py"]
