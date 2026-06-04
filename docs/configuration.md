# Configuration

The backend is configured through two complementary mechanisms: a **`config.yaml`** file for
static, version-controlled settings, and **environment variables** for secrets and
deployment-specific values.

---

## `config.yaml`

Located at `backend/scheduler/config.yaml`, this file is loaded at startup via
[OmegaConf](https://omegaconf.readthedocs.io/) and is the primary place to tune scheduler
behaviour without touching code.

```yaml
collector:
  observation_classes: [SCIENCE, PROGCAL, PARTNERCAL]
  program_types: [Q, LP, FT, DD]
  time_slot_length: 1.0        # minutes
  with_redis: false
  parallel_viscalc: false
  # "sight" reads pre-computed visibility from the Sight service and falls back
  # to the local joblib calculator on transport failure.
  # "local" always uses the in-process calculator.
  visibility_strategy: sight

optimizer:
  name: GREEDYMAX

selector:
  buffer_type: FLAT_MINUTES
  buffer_amount: 30.0          # minutes

server:
  port: 8000
  host: 0.0.0.0

app:
  external_ephemerides: false

sight:
  api_url: http://localhost:9800/api/v1
  timeout: 300                 # seconds per request
```

### Key settings

| Key | Default | Description |
|---|---|---|
| `collector.observation_classes` | `[SCIENCE, PROGCAL, PARTNERCAL]` | Observation classes included in scheduling. |
| `collector.program_types` | `[Q, LP, FT, DD]` | Program types considered by the collector. |
| `collector.time_slot_length` | `1.0` | Time slot granularity in minutes. |
| `collector.with_redis` | `false` | Enable Redis-backed caching for collector data. |
| `collector.parallel_viscalc` | `false` | Run visibility calculations in parallel. |
| `collector.visibility_strategy` | `sight` | `sight` (remote service) or `local` (in-process). |
| `optimizer.name` | `GREEDYMAX` | Scheduling algorithm to use. |
| `selector.buffer_type` | `FLAT_MINUTES` | Type of time buffer added around observations. |
| `selector.buffer_amount` | `30.0` | Size of the buffer in minutes. |
| `server.host` / `server.port` | `0.0.0.0:8000` | Uvicorn bind address. Overridden by `PORT` env var on Heroku. |
| `app.external_ephemerides` | `false` | Use external (CloudCube S3) ephemeris data instead of bundled data. |
| `sight.api_url` | `http://localhost:9800/api/v1` | Base URL of the Sight visibility service. |
| `sight.timeout` | `300` | HTTP timeout (seconds) for Sight requests. |

---

## Environment variables

Secrets and deployment-specific values are passed as environment variables. For local
development, place them in a `backend/.env` file (never commit this file).

```sh
# backend/.env  — local development only
SCHEDULER_MODE=VALIDATION
REDISCLOUD_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:password@localhost:5432/scheduler
```

### Reference

| Variable | Required | Description                                                                                                                                                                      |
|---|----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `SCHEDULER_MODE` | **Yes**  | Controls the runtime mode. See [Scheduler modes](#scheduler-modes) below.                                                                                                        |
| `PORT` | No       | HTTP port for the server. Heroku sets this automatically; falls back to `server.port` in `config.yaml`.                                                                          |
| `REDISCLOUD_URL` | No       | Redis connection URL (e.g. `redis://localhost:6379`). Required when `collector.with_redis: true`.                                                                                |
| `DATABASE_URL` | No       | PostgreSQL connection URL for the Sight service database. Heroku Postgres sets this automatically; the `postgres://` scheme is normalised to `postgresql+asyncpg://` at startup. |
| `DB_POOL_SIZE` | No       | SQLAlchemy connection pool size (default `5`, max `20`).                                                                                                                         |
| `DB_POOL_OVERFLOW` | No       | SQLAlchemy max overflow connections (default `10`, max `20`).                                                                                                                    |
| `DB_ECHO_SQL` | No       | Set to `true` to log all SQL statements (default `false`).                                                                                                                       |
| `CLOUDCUBE_URL` | No       | S3-compatible bucket URL for ephemeris cube storage. Required when `app.external_ephemerides: true`.                                                                             |
| `CLOUDCUBE_ACCESS_KEY_ID` | No       | AWS/CloudCube access key. Required together with `CLOUDCUBE_URL`.                                                                                                                |
| `CLOUDCUBE_SECRET_ACCESS_KEY` | No       | AWS/CloudCube secret key. Required together with `CLOUDCUBE_URL`.                                                                                                                |
| `GPP_TOKEN` | **Yes**  | Allows the Scheduler to connect to other GPP services like the ODB. If the development client is used add `GPP_DEVELOPMENT_TOKEN` instead                                        |
| `WEATHER_URL` | **Yes** | Connects with the Weather service. Can be a Heroku URL or a local one set by Docker                                                                                              |
|`VITE_WEATHER_URL` | **Yes**| Same as above but used by the frontend app |

### Scheduler modes

`SCHEDULER_MODE` determines how the scheduler behaves at startup:

| Value | Mode | Description |
|---|---|---|
| `REALTIME` | Operation | Connects to live GPP data and runs in real-time. |
| `SIMULATION` | Simulation | Runs scheduling simulations over historical or synthetic data. |
| `VALIDATION` | Validation | Default fallback. Validates plans without live data connections. |

!!! note
    If `SCHEDULER_MODE` is unset or invalid, the scheduler falls back to `VALIDATION` mode automatically.

---

## Setting up for local development

1. Copy the example below into `backend/.env` and fill in the values for your setup:

    ```sh
    SCHEDULER_MODE=VALIDATION
    # Optional — only needed with collector.with_redis: true
    REDISCLOUD_URL=redis://localhost:6379
    # Optional — only needed for the Sight service
    DATABASE_URL=postgresql://user:password@localhost:5432/scheduler
    ```

2. Start the backend:

    ```sh
    cd backend
    uv run python -m scheduler.main
    ```

    The server listens on `http://localhost:8000` by default (configurable via `server.port` in `config.yaml`).

3. To switch modes, change `SCHEDULER_MODE` in `.env` or export it in your shell:

    ```sh
    SCHEDULER_MODE=SIMULATION uv run python -m scheduler.main
    ```
