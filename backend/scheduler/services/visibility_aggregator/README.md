# Visibility aggregator

Background service that keeps the **Sight DB** current for the programs GPP
reports as *READY*. For the current semester it ensures:

- **Stage 1** (`target_night_data`) exists for every **sidereal** target, and
- **Stage 2** (`visibility_data`) exists for every observation,

adding only what is missing. It reuses the existing `Calculator` (the same
engine `scripts/fill_sight.py` drives), so this package is orchestration + a DB
diff, not new visibility math.

> **Scope:** sidereal targets only for now. Non-sidereal (Horizons) targets are
> skipped; see the follow-up note in the code.

## Running it

It is a one-off process, intended for **Heroku Scheduler**:

```
cd /home && uv run python -m scheduler.services.visibility_aggregator.runner
```

Heroku Scheduler's minimum interval is 1 hour; overlapping ticks are
prevented by the coordination row (a second run exits if one is already active).
Run it locally / manually the same way against a populated `DATABASE_URL`.

### Heroku setup (env vars)

A Heroku Scheduler job is a **one-off dyno in the same app**, so it automatically
inherits all of the app's config vars — there is no separate env config for
scheduler jobs. The only requirement is that the relevant vars are set on the app
(the same one running the web/operation dyno):

The aggregator's own knobs live in `config.yaml` (see **Tuning** below), not in
env vars — only secrets/connection info come from the environment. `CLOUDCUBE_*`
/ ephemerides are **not** required either: those are only for non-sidereal
Horizons targets, which this runner skips for now.


## Coordination (interlock)

Two rows in `scheduler_coordination`, shared with the always-on operation dyno:

- `visibility_aggregator` — set active while a run is in progress. Before
  creating a plan, `EngineRT.compute_event_plan` calls
  `coordination.wait_until_aggregator_idle()` and **blocks until the run
  finishes** (the hard interlock).
- `night_execution` — set active while the operation process is computing a
  plan, so a cron tick won't start concurrently.

A run is **skipped** when a night is in progress (12° dark time at GN or GS,
computed astronomically) or a plan is being computed. Liveness is a committed
`heartbeat_at` plus a staleness threshold, so a crashed holder never wedges the
interlock.

## Tuning (config.yaml)

All knobs live under the `visibility_aggregator` section of
`scheduler/config.yaml`:

- `plan_wait_timeout` (seconds) — cap how long plan creation blocks for the
  aggregator. `null` = block until finished. Env-overridable per-deploy via
  `VIS_AGG_PLAN_WAIT_TIMEOUT` (a safety valve if a long first-time backfill must
  not stall a night).
- `stale_after_seconds` — a coordination holder whose heartbeat is older than
  this is treated as dead (so a crashed run never wedges the interlock).
- `heartbeat_interval_seconds` — how often a run refreshes its heartbeat.
- `idle_poll_interval_seconds` — how often the operation process polls while
  blocked on a run.
- `target_batch_size` — targets per committed Stage-1 batch.

## Long runs, timeouts, and failures

These three numbers are independent — `stale_after_seconds` is **not** a run
timeout:

- **`heartbeat_interval_seconds` (60s)** — a live run refreshes `heartbeat_at`
  this often. The parse phase and each compute batch yield to the event loop, so
  the heartbeat keeps firing even during a multi-minute backfill.
- **`stale_after_seconds` (600s)** — how long *readers* wait without a heartbeat
  before treating the holder as **dead**. It only matters if a run crashes; a
  healthy run heartbeats 10× within this window and may run as long as it needs
  (a ~30 min semester-start backfill is fine — the Heroku one-off/Scheduler dyno
  has no such limit).
- **`plan_wait_timeout` (null)** — how long the operation process blocks before
  planning anyway. `null` = block until the run finishes. If a run *crashes*,
  `stale_after_seconds` is the safety net: planning resumes within 600s instead
  of blocking forever.

**Resumability.** The work is committed per Stage-1 batch and per Stage-2 night,
and every write is an idempotent upsert against the "what's already there" diff.
So if a dyno is killed (cycling, deploy, SIGTERM, dropped connection), nothing is
half-written: the next cron tick simply fills the remaining gaps. On `SIGTERM`
the run cancels gracefully and releases the coordination row immediately;
otherwise the row expires via `stale_after_seconds`.

**Partial data is safe.** The scheduler only ever reads whatever Stage-1/Stage-2
rows exist and skips the rest (its pre-existing behaviour). An incomplete run
just means fewer gaps filled this tick, not corrupt or inconsistent data.

## Observability

`query { visibilityAggregatorStatus { active stale holder heartbeatAt detail } }`
reports the current run state.