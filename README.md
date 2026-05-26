# GPP Scheduler

> Automatic scheduling service at Gemini International Observatory

![Python](https://img.shields.io/badge/python-3.11+-blue)

## Development

### Environment variables

To run a local development version first make sure your environment have the appropriate variables already set. This will change depending on the shell you are using. In bash and zsh you can copy the .env.example file to a new .env file, fill the secrets in this new .env file and run

```bash
source .env
```

Also the `PYTHONPATH` environment variable should be properly set to add the `backend` directory of this repository.

### Backend

```bash
cd backend
uv sync --group gpp-dev --no-group gpp-prod
uv run python scheduler/main.py
```

### Frontend

```bash
pnpm install
pnpm frontend dev
```

### Docs

```bash
uv sync --group docs
uv run mkdocs serve
```

## Local Deployment

The full stack runs via Docker Compose: two backend instances (REALTIME on port 8000, VALIDATION on port 8001) and a React frontend on port 80.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) with Compose v2
- [Git LFS](https://git-lfs.com/) — required for the ephemerides data

```bash
git lfs install
git lfs pull
```

### Setup

1. Copy the environment file and fill in any secrets:

```bash
cp .env.example .env
```

2. Build and start all services:

```bash
docker compose build
docker compose up
```

| Service              | URL                           | Mode         |
| -------------------- | ----------------------------- | ------------ |
| Frontend             | http://localhost              | —            |
| Backend (Validation) | http://localhost:8001/graphql | `VALIDATION` |
| Backend (Realtime)   | http://localhost:8000/graphql | `REALTIME`   |
| Weather              | http://localhost:4000         | —            |

### Optional: Redis

To enable Redis caching, also set `collector.with_redis: true` in `backend/scheduler/config.yaml`, then:

```bash
REDISCLOUD_URL=redis://redis:6379 docker compose --profile redis up
```

### Updating the GraphQL schema

When the backend schema changes, regenerate `backend/scheduler.graphql` before rebuilding:

```bash
uv run python scripts/export_graphql_schema.py
docker compose build frontend
```
