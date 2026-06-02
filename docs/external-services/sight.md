# Sight: Visibility Service

Sight is an in-process visibility engine that pre-computes and caches per-target,
per-site, per-night observability data. By storing results in a PostgreSQL database,
the Scheduler avoids re-running expensive trigonometric calculations on every
scheduling cycle.

---

## How it works

Sight uses a **two-stage pipeline**:

```mermaid
graph LR
    A[Target + Site + Night] --> B[Stage 1\nPositions]
    B --> C[Stage 2\nVisibility]
    C --> D[(PostgreSQL)]
    D --> E[Scheduler]
```

| Stage | What it computes | Stored arrays |
|---|---|---|
| **Stage 1** | Target ephemerides for every time slot of the night | RA, Dec, altitude, azimuth, hour angle, airmass, parallactic angle |
| **Stage 2** | Observation-level visibility given constraints | `remaining_minutes`, `visible_ranges` (boolean mask → index ranges) |

Stage 1 data is reused across all observations that share a target. Stage 2 is
computed per observation because it depends on individual constraints (sky
background, elevation limits, timing windows).

---

## Configuration

### Database connection

Sight uses PostgreSQL. Set `DATABASE_URL` in `backend/.env`:

```sh
DATABASE_URL=postgresql://user:password@localhost:5432/scheduler
```

See the [Configuration](../configuration.md#environment-variables) page for the full
list of `DB_*` tuning knobs (`DB_POOL_SIZE`, `DB_POOL_OVERFLOW`, `DB_ECHO_SQL`).

To use a Heroku remote Sight endpoint instead you can ask the correct url in the Slack channel for the Scheduler.

---

## Local development

The database starts empty. Run the seeding script **once** to pre-populate it
with visibility data for the bundled OCS validation programs
(GN · 2018-08-01 → 2019-01-31):

```sh
cd backend
DATABASE_URL=postgresql://user:password@localhost:5432/scheduler \
  uv run python -m scheduler.scripts.fill_sight
```

The script is **idempotent** — re-running it skips targets that already exist
and upserts any missing Stage 1 / Stage 2 rows without touching existing data.

!!! note "Skipping local Sight altogether"
    If you don't need pre-computed data locally, switch the visibility strategy
    to `local` in `config.yaml`. The Scheduler will run the in-process
    calculator directly, with no database required:

    ```yaml
    collector:
      visibility_strategy: local
    ```
---

## Calculator API

`Calculator` is the internal Python interface to Sight.
All scheduler code that needs visibility data goes through this class.
It is instantiated per-request via SQLAlchemy's `AsyncSession`.

```python
from scheduler.services.sight.calculator.calculator import Calculator
from scheduler.services.sight.database.connection import session_scope

async with session_scope() as session:
    calc = Calculator(session)
    result = await calc.calculate_visibility(requests, night_date)
```

### Data models

#### `TargetCreate`

Used when registering a new target with Sight.

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | `str` | Yes | Unique target name. |
| `is_sidereal` | `bool` | No (default `True`) | `False` for non-sidereal targets. |
| `base_ra` | `float` | Yes | Right ascension in degrees `[0, 360)`. |
| `base_dec` | `float` | Yes | Declination in degrees `[-90, 90]`. |
| `pm_ra` | `float \| None` | No | Proper motion in RA (mas/yr). |
| `pm_dec` | `float \| None` | No | Proper motion in Dec (mas/yr). |
| `epoch` | `float \| None` | No (default `2000.0`) | Coordinate epoch. |
| `horizons_id` | `str \| None` | No | JPL Horizons ID for non-sidereal targets. |
| `tag` | `str \| None` | No | Non-sidereal type: `majorbody`, `asteroid`, or `comet`. |

#### `ObservationConstraints`

Observation-level constraints fed into Stage 2.

| Field | Type | Default | Description |
|---|---|---|---|
| `target_sb` | `float` | `1.0` | Sky background fraction `(0.2, 0.5, 0.8, 1.0)`. |
| `elevation_type` | `ElevationType` | `airmass` | `none`, `hour_angle`, or `airmass`. |
| `elevation_min` | `float` | `1.0` | Lower bound for the chosen elevation type. |
| `elevation_max` | `float` | `2.05` | Upper bound for the chosen elevation type. |
| `timing_windows` | `list[TimingWindow]` | `[]` | UTC time windows when the observation may execute. |
| `has_resources` | `bool` | `True` | Whether required instrument resources are available. |
| `can_schedule` | `bool` | `True` | Whether the observation is schedulable. |

#### `ObservationRequest`

The unit of work sent to the Calculator.

| Field | Type | Description |
|---|---|---|
| `observation_id` | `str` | Unique observation identifier. |
| `target_name` | `str` | Must match a name registered via `create_targets_bulk`. |
| `site_id` | `str` | `"GN"` (Gemini North) or `"GS"` (Gemini South). |
| `constraints` | `ObservationConstraints` | Observing constraints. |

#### `VisibilityResult`

Returned per observation by most read methods.

| Field | Type | Description |
|---|---|---|
| `observation_id` | `str` | Mirrors the request. |
| `target_name` | `str` | Target name. |
| `site` | `str` | `"GN"` or `"GS"`. |
| `night_date` | `date` | The night this result is valid for. |
| `remaining_minutes` | `int` | Total schedulable minutes on this night. |
| `visible_ranges` | `list[list[int]]` | Time-slot index ranges where the target is visible, e.g. `[[10, 45], [120, 180]]`. |

---

### Methods

#### `create_targets_bulk`

Register targets and pre-compute Stage 1 data over a date range.

```python
response: BulkTargetCreateResponse = await calc.create_targets_bulk(
    targets=[TargetCreate(name="NGC 1234", is_sidereal=True, base_ra=50.0, base_dec=-30.0)],
    start_date=date(2025, 2, 1),
    end_date=date(2025, 3, 31),
)
# response.created  → number of new targets
# response.failed   → number of skipped/failed targets
# response.errors   → list of error messages
```

Already-existing targets are skipped without error.

---

#### `precompute_stage1`

Compute Stage 1 arrays for existing targets across a date range.
Useful after adding new nights or sites without re-creating targets.

```python
result = await calc.precompute_stage1(
    start_date=date(2025, 2, 1),
    end_date=date(2025, 2, 28),
    target_names=["NGC 1234", "HD 12345"],   # None → all targets
    site_ids=["GN"],                          # None → GN + GS
)
# result → {"targets": 2, "sites": 1, "nights": 28, "total_computations": 56}
```

---

#### `calculate_visibility`

On-demand Stage 2 calculation for a single night. Results are **not** persisted.

```python
response: CalculationResponse = await calc.calculate_visibility(
    requests=[
        ObservationRequest(
            observation_id="GN-2025A-Q-1-1",
            target_name="NGC 1234",
            site_id="GN",
            constraints=ObservationConstraints(target_sb=0.5, elevation_min=1.0, elevation_max=2.0),
        )
    ],
    night_date=date(2025, 2, 15),
)
for result in response.results:
    print(result.observation_id, result.remaining_minutes, result.visible_ranges)
```

---

#### `store_visibility`

Calculate **and persist** Stage 2 results across a date range. Used by
`fill_sight.py` to seed the database.

```python
stats = await calc.store_visibility(
    requests=requests,
    start_date=date(2025, 2, 1),
    end_date=date(2025, 3, 31),
)
# stats → {"stored": 840, "nights": 59}
```

---

#### `get_visible_observations`

Return only observations that are visible on a given night.
Checks stored Stage 2 data first; falls back to on-demand calculation for
observations without stored results.

```python
visible: list[VisibilityResult] = await calc.get_visible_observations(
    requests=requests,
    night_date=date(2025, 2, 15),
)
```

---

#### `get_precalculated_visibility`

Fetch stored Stage 2 rows filtered by observation or target over a date range.

```python
results = await calc.get_precalculated_visibility(
    start_date=date(2025, 2, 1),
    end_date=date(2025, 2, 28),
    observation_id="GN-2025A-Q-1-1",  # optional
    target_name="NGC 1234",            # optional; at least one filter required
)
```

---

#### `get_precalculated_visibility_bulk`

Batch fetch stored Stage 2 rows for many observations, grouped by observation → target → night.

```python
data = await calc.get_precalculated_visibility_bulk(
    observation_ids=["GN-2025A-Q-1-1", "GN-2025A-Q-2-1"],
    start_date=date(2025, 2, 1),
    end_date=date(2025, 2, 28),
)
# data["GN-2025A-Q-1-1"]["targets"]["NGC 1234"]["nights"]["2025-02-15"]
# → {"night_date": ..., "site": "GN", "remaining_minutes": 120, "visible_ranges": [[10, 45]]}
```

---

#### `get_cumulative_remaining_visibility`

Sum `remaining_minutes` across all nights in a range, grouped by observation → target.

```python
data = await calc.get_cumulative_remaining_visibility(
    observation_ids=["GN-2025A-Q-1-1"],
    start_date=date(2025, 2, 1),
    end_date=date(2025, 2, 28),
)
# data["GN-2025A-Q-1-1"]["targets"]["NGC 1234"]
# → {"site": "GN", "cumulative_remaining_minutes": 1680, "nights_with_visibility": 14}
```

---

#### `get_stage1_bulk`

Retrieve full Stage 1 arrays (all seven fields) for a set of targets, sites, and nights.

```python
data = await calc.get_stage1_bulk(
    target_names=["NGC 1234"],
    site_ids=["GN"],
    start_date=date(2025, 2, 1),
    end_date=date(2025, 2, 28),
)
# data["NGC 1234"]["nights"]["GN_2025-02-15"]
# → {"night_date", "site", "night_duration_minutes",
#    "ra"[], "dec"[], "alt"[], "az"[], "hourangle"[], "airmass"[], "par_ang"[]}
# Angular arrays are in radians.
```

---

#### `get_stage1_greedymax_bulk`

Same as `get_stage1_bulk` but returns only the fields needed by the GreedyMax
optimizer (RA, Dec, alt, az, airmass, hour angle — no parallactic angle),
reducing payload size.

```python
data = await calc.get_stage1_greedymax_bulk(
    target_names=["NGC 1234"],
    site_ids=["GN", "GS"],
    start_date=date(2025, 2, 1),
    end_date=date(2025, 2, 28),
)
```
