# Scheduler scripts

### `aggregate_parallel.py`
Local-only, parallel version of one visibility-aggregator pass: parses the ODB program set with the
aggregator's own code, then fills night events, Stage 1 and Stage 2 in the Sight DB from a process pool
(one DB connection per worker, bulk upserts). Writes the same rows as the aggregator, ~8x faster on a
14-core laptop. Safe to re-run or interrupt; it resumes from what is missing. Does not touch the ODB
change watermark or the coordination interlock.

```
cd backend
DATABASE_URL=postgresql+asyncpg://scheduler:scheduler_dev@localhost:5433/scheduler GPP_TOKEN=... \
    python -m scheduler.scripts.aggregate_parallel [--workers N] [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--dry-run]
```

### `run_greedymax.py`
Run the Scheduler event loop, using GreedyMax to schedule visits during a configurable number of nights
with the desired visibility calculations.

### `download_programs.py`
Downloads program needed for the Scheduler to work. 
DEFAULT_PROGRAMS variable specifies the list of programs the script is going to download
Programs (the .json.gz file) needs to be on the `data/` for the Provider to work properly.

### `odb.extractor_atoms.py`
Script that generates atom for OCS programs. This is Bryan's experimental work and does not use current
mini-model structures. 