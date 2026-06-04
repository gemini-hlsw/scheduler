# Visibility aggregator

Background service that keeps the **Sight DB** current for the programs GPP
reports as *available*. For the current semester it ensures:

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

Heroku Scheduler's minimum interval is 10 minutes; overlapping ticks are
prevented by the coordination row (a second run exits if one is already active).
Run it locally / manually the same way against a populated `DATABASE_URL`.

### Heroku setup (env vars)

A Heroku Scheduler job is a **one-off dyno in the same app**, so it automatically
inherits all of the app's config vars — there is no separate env config for
scheduler jobs. The only requirement is that the relevant vars are set on the app
(the same one running the web/operation dyno):

| Var | Source | Needed because |
| --- | --- | --- |
| `DATABASE_URL` | Heroku Postgres add-on (auto) | shared Sight DB + coordination. Must be the same DB as the web dyno. `postgres://` is normalised to `postgresql+asyncpg://` in `sight/config.py`. |
| GPP credentials (e.g. `GPP_DEVELOPMENT_TOKEN`) | already set for the web dyno | the runner calls GPP. If the web app talks to GPP today, the cron inherits the same credential. |
| `VIS_AGG_PLAN_WAIT_TIMEOUT` (optional) | you | caps how long plan creation blocks for a run (default: block until finished). |

`CLOUDCUBE_*` / ephemerides are **not** required — those are only for
non-sidereal Horizons targets, which this runner skips for now.

```bash
# 1. one-time: add the add-on
heroku addons:create scheduler:standard -a <app>

# 2. confirm the inherited config (DATABASE_URL + GPP creds present)
heroku config -a <app>

# 3. (optional) safety valve / set any missing var
heroku config:set VIS_AGG_PLAN_WAIT_TIMEOUT=900 -a <app>

# 4. add the job in the dashboard (command + interval: every 10 min / hourly / daily)
heroku addons:open scheduler -a <app>
#    Command:   cd /home && uv run python -m scheduler.services.visibility_aggregator.runner
#    Dyno size: pick Standard-2X/Performance for the heavy first full-semester backfill.

# 5. smoke-test once on a one-off dyno (also inherits config vars)
heroku run "cd /home && uv run python -m scheduler.services.visibility_aggregator.runner" -a <app>
```

The `scheduler_coordination` table is created by the `009` Alembic migration,
which runs automatically in the existing `heroku.yml` release phase on deploy.

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

## Tuning (env vars)

- `VIS_AGG_PLAN_WAIT_TIMEOUT` (seconds) — cap how long plan creation blocks for
  the aggregator. Default: unset = block until finished. Set this as an
  operational safety valve if a long first-time backfill must not stall a night.

## Observability

`query { visibilityAggregatorStatus { active stale holder heartbeatAt detail } }`
reports the current run state.

> **First run** is heavy (all sidereal targets × 2 sites × ~6 months). Every
> later tick is cheap because it only fills gaps. Consider running the first
> backfill in a daytime window (or manually) so it can't block the start of a
> night.
