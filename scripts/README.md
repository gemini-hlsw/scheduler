### `scripts/stress_realtime.py` (repo root)
Realtime stress test: times `EngineRT.compute_event_plan` with up to 1500 observations per site, built by
cloning the programs active today (`--source odb` uses live ODB data instead). Needs GPP credentials in
the environment and runs from the repo root. Results go to `stress/results/report.md`, one section per
`--label`; set `OTEL_EXPORTER_OTLP_ENDPOINT` to also send the metrics to Grafana.

```
# Smoke test, local visibility
uv run python scripts/stress_realtime.py run --sizes 25 --plans 1 --visibility local --label smoke

# Sight visibility: seed the local DB first, then run with the same --night, --cap and --programs
DATABASE_URL=postgresql+asyncpg://scheduler:scheduler_dev@localhost:5433/scheduler \
    uv run python scripts/stress_realtime.py seed-sight --window full
DATABASE_URL=postgresql+asyncpg://scheduler:scheduler_dev@localhost:5433/scheduler \
    uv run python scripts/stress_realtime.py run --plans 3 --visibility sight --label sight
```
`uv run python scripts/stress_realtime.py run --help` lists the rest (`--sizes`, `--vis-end`, `--offset-hours`, `--iq`, `--cc`).
