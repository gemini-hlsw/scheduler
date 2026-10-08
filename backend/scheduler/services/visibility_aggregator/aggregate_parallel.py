#!/usr/bin/env python3
# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""Fill the Sight DB from the aggregator's ODB program set, in parallel. Local only.
"""

import argparse
import asyncio
import multiprocessing
import os
import pickle
import signal
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, as_completed, wait
from dataclasses import dataclass, fields
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from typing import NamedTuple, Optional

from lucupy.minimodel import Semester
from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini.geminiproperties import GeminiProperties
from sqlalchemy import select

from scheduler.clients.gpp import gpp
from scheduler.services import logger_factory
from scheduler.services.horizons import horizons_session
from scheduler.services.sight.calculations.stage1 import (
    _HorizonsSiteAdapter,
    _HorizonsTargetAdapter,
)
from scheduler.services.sight.calculator.calculator import Calculator
from scheduler.services.sight.calculator.constants import SITE_KEY_TO_ID
from scheduler.services.sight.calculator.models import ObservationRequest, TargetCreate
from scheduler.services.sight.database.models import NightEvent
from scheduler.services.sight.database.connection import (
    dispose_engine,
    init_db_engine,
    session_scope,
)
from scheduler.services.visibility_aggregator.aggregator import (
    _collect_requests,
    _format_duration,
    progress_eta_seconds,
    target_windows,
)
from scheduler.services.visibility_aggregator.memory_guard import MemoryGuard

_logger = logger_factory.create_logger(__name__, with_id=False)

# Targets per task.
DEFAULT_CHUNK_SIZE = 200

_PROGRESS_EVERY_SECONDS = 10.0

_HORIZONS_PROGRESS_EVERY = 25


@dataclass(frozen=True)
class TargetSnapshot:
    """The Target columns Stage 1 reads, detached from any session.

    Pickled into each worker once at start-up instead of re-queried per task.
    """
    id: int
    name: str
    is_sidereal: bool
    base_ra: Optional[float]
    base_dec: Optional[float]
    pm_ra: Optional[float]
    pm_dec: Optional[float]
    epoch: Optional[float]
    horizons_id: Optional[str]
    tag: Optional[str]
    updated_at: datetime

    @classmethod
    def of(cls, target) -> "TargetSnapshot":
        return cls(**{f.name: getattr(target, f.name) for f in fields(cls)})


class Plan(NamedTuple):
    targets: dict[str, TargetSnapshot]
    requests_by_target: dict[str, list[ObservationRequest]]
    # Per observation: the nights its program is active for.
    windows: dict[str, tuple[date, date]]
    # (site_id, night) pairs with no night events yet.
    missing_night_events: list[tuple[int, date]]
    # (night, target names) work units for sidereal targets, which run first.
    chunks: list[tuple[date, tuple[str, ...]]]
    # Non-sidereal work units, queued once their Horizons files are cached.
    deferred_chunks: list[tuple[date, tuple[str, ...]]]
    first: date
    last: date


class ChunkResult(NamedTuple):
    stage1_rows: int
    stage2_rows: int


class EphemeridesResult(NamedTuple):
    ready: int
    failed: list[str]
    seconds: float


def plan_chunks(
    windows_by_target: dict[str, tuple[date, date]],
    first: date,
    last: date,
    chunk_size: int,
) -> list[tuple[date, tuple[str, ...]]]:
    """Cut every night in [first, last] into chunks of the targets due that night.

    A target is due on the nights of its window, so each (target, night) lands
    in exactly one chunk and no two tasks ever write the same rows.
    """
    chunks: list[tuple[date, tuple[str, ...]]] = []
    night = first
    while night <= last:
        due = sorted(
            name for name, (start, end) in windows_by_target.items()
            if start <= night <= end
        )
        for offset in range(0, len(due), chunk_size):
            chunks.append((night, tuple(due[offset:offset + chunk_size])))
        night += timedelta(days=1)
    return chunks


async def _ensure_targets(
    payloads: dict[str, TargetCreate],
) -> dict[str, TargetSnapshot]:
    """Create the target rows Sight is missing and snapshot all of them.

    Same columns ``Calculator.create_targets_bulk`` writes; Stage 1 is left to
    the workers.
    """
    async with session_scope() as session:
        repo = Calculator(session).target_repo
        found = await repo.get_by_names(list(payloads))
        created = 0
        for name, payload in payloads.items():
            if name in found:
                continue
            found[name] = await repo.create(
                name=payload.name,
                is_sidereal=payload.is_sidereal,
                base_ra=payload.base_ra,
                base_dec=payload.base_dec,
                pm_ra=payload.pm_ra,
                pm_dec=payload.pm_dec,
                epoch=payload.epoch,
                horizons_id=payload.horizons_id,
                tag=payload.tag,
            )
            created += 1
    _logger.info(f"Targets: {created} created, {len(found) - created} already in Sight.")
    return {name: TargetSnapshot.of(target) for name, target in found.items()}


async def _find_missing_night_events(nights: list[date]) -> list[tuple[int, date]]:
    async with session_scope() as session:
        calc = Calculator(session)
        missing = []
        for site_id in sorted(await calc.get_sites()):
            existing = await calc.night_repo.get_existing_dates(
                site_id, nights[0], nights[-1]
            )
            missing += [(site_id, night) for night in nights if night not in existing]
    return missing


async def build_plan(
    payloads: dict[str, TargetCreate],
    requests: list[ObservationRequest],
    windows: dict[str, tuple[date, date]],
    *,
    start: Optional[date] = None,
    end: Optional[date] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    dry_run: bool = False,
) -> Optional[Plan]:
    """Turn the parsed program set into work units. None if there is nothing to do.

    Needs an initialised DB engine. Writes the missing target rows unless
    ``dry_run``.
    """
    windows_by_target = target_windows(requests, windows)
    if not windows_by_target:
        _logger.info("No targets with a program window; nothing to do.")
        return None
    first = max(start or date.min, min(w[0] for w in windows_by_target.values()))
    last = min(end or date.max, max(w[1] for w in windows_by_target.values()))
    if first > last:
        _logger.info(f"No program window overlaps {start}..{end}; nothing to do.")
        return None

    # Non-sidereal targets read Horizons files that may not be cached yet, so
    # their chunks wait while one worker downloads the files and the rest
    # compute the sidereal chunks.
    sidereal = {
        name: window for name, window in windows_by_target.items()
        if payloads[name].is_sidereal
    }
    nonsidereal = {
        name: window for name, window in windows_by_target.items()
        if name not in sidereal
    }
    chunks = plan_chunks(sidereal, first, last, chunk_size)
    deferred_chunks = plan_chunks(nonsidereal, first, last, chunk_size)
    nights = sorted({night for night, _ in chunks + deferred_chunks})
    missing_night_events = await _find_missing_night_events(nights)
    target_nights = sum(len(names) for _, names in chunks + deferred_chunks)
    _logger.info(
        f"Plan: {first}..{last}, {len(windows_by_target)} targets "
        f"({len(nonsidereal)} non-sidereal), {len(requests)} observations, "
        f"{target_nights} target-nights in {len(chunks) + len(deferred_chunks)} "
        f"chunks; {len(missing_night_events)} night events missing."
    )
    if dry_run:
        return None

    requests_by_target: dict[str, list[ObservationRequest]] = {}
    for request in requests:
        if request.observation_id in windows:
            requests_by_target.setdefault(request.target_name, []).append(request)
    targets = await _ensure_targets(
        {name: payloads[name] for name in windows_by_target}
    )
    return Plan(
        targets, requests_by_target, windows, missing_night_events,
        chunks, deferred_chunks, first, last,
    )


# --- worker side ---------------------------------------------------------------

# Per-process state set by _init_worker: the event loop that owns this
# worker's DB engine, plus the read-only program set every task looks into.
_worker: dict = {}


def _init_worker(program_set_path: str, stop) -> None:
    # Ctrl-C belongs to the parent, which cancels the queue and lets running
    # tasks commit or roll back cleanly instead of dying mid-write. ``stop``
    # is how it tells the long Horizons download to quit early.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    with open(program_set_path, "rb") as program_set:
        targets, requests_by_target, windows = pickle.load(program_set)
    ObservatoryProperties.set_properties(GeminiProperties)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(init_db_engine())
    _worker.update(
        loop=loop,
        stop=stop,
        targets=targets,
        requests_by_target=requests_by_target,
        windows=windows,
    )


def fill_night_events(site_id: int, night: date) -> None:
    _worker["loop"].run_until_complete(_fill_night_events(site_id, night))


async def _fill_night_events(site_id: int, night: date) -> None:
    async with session_scope() as session:
        calc = Calculator(session)
        site = (await calc.get_sites())[site_id]
        await calc._calculate_and_store_night_events(site, night)


def fill_ephemerides(first: date, last: date) -> EphemeridesResult:
    return _worker["loop"].run_until_complete(_fill_ephemerides(first, last))


async def _fill_ephemerides(first: date, last: date) -> EphemeridesResult:
    """Cache every Horizons file the non-sidereal chunks are going to read.

    Files are per (site, target, semester of the night's sunset). Each one is
    requested through the same call Stage 1 makes for the first night that
    needs it, so the file cached is exactly the one Stage 1 reads. Downloads
    go one at a time through the Horizons client's cache lock.
    """
    t0 = time.perf_counter()
    async with session_scope() as session:
        sites = await Calculator(session).get_sites()
        rows = (await session.execute(
            select(NightEvent.site_id, NightEvent.night_date, NightEvent.sunset, NightEvent.sunrise)
            .where(NightEvent.night_date.between(first, last))
        )).all()
    # Naive UTC, as Stage 1 hands them to the Horizons client.
    night_times = {
        (site_id, night): (
            sunset.astimezone(timezone.utc).replace(tzinfo=None),
            sunrise.astimezone(timezone.utc).replace(tzinfo=None),
        )
        for site_id, night, sunset, sunrise in rows
    }
    semesters = {
        key: Semester.find_semester_from_date(sunset)
        for key, (sunset, _) in night_times.items()
    }

    requests = [r for rs in _worker["requests_by_target"].values() for r in rs]
    calls: dict[tuple[str, int, Semester], tuple[datetime, datetime]] = {}
    for name, (start, end) in sorted(target_windows(requests, _worker["windows"]).items()):
        target = _worker["targets"][name]
        if target.is_sidereal or not target.horizons_id:
            continue
        night = max(start, first)
        while night <= min(end, last):
            for site_id in sites:
                calls.setdefault(
                    (name, site_id, semesters[(site_id, night)]),
                    night_times[(site_id, night)],
                )
            night += timedelta(days=1)

    failed = []
    for done, ((name, site_id, semester), (sunset, sunrise)) in enumerate(calls.items(), start=1):
        if _worker["stop"].is_set():
            break
        try:
            with horizons_session(_HorizonsSiteAdapter(sites[site_id]), sunset, sunrise, 1) as hs:
                hs.get_ephemerides(_HorizonsTargetAdapter(_worker["targets"][name]))
        except Exception as exc:
            failed.append(f"{name} at {sites[site_id].name} for {semester}: {exc!r}")
        if done % _HORIZONS_PROGRESS_EVERY == 0:
            _logger.info(f"Horizons: {done}/{len(calls)} ephemeris files checked or downloaded.")
    return EphemeridesResult(len(calls) - len(failed), failed, time.perf_counter() - t0)


def fill_chunk(night: date, names: tuple[str, ...]) -> ChunkResult:
    return _worker["loop"].run_until_complete(_fill_chunk(night, names))


async def _fill_chunk(night: date, names: tuple[str, ...]) -> ChunkResult:
    """Stage 1 then Stage 2 for one night's chunk of targets, one transaction.

    The math is the Calculator's own (``_stage1_row``, ``_calculate_stage2``),
    so rows match what the aggregator would store.
    """
    targets = [_worker["targets"][name] for name in names]
    windows = _worker["windows"]

    async with session_scope() as session:
        calc = Calculator(session)
        sites = await calc.get_sites()
        night_events = {
            ne.site_id: ne
            for ne in await calc.night_repo.get_for_multiple_sites(list(sites), night)
        }
        if len(night_events) != len(sites):
            raise RuntimeError(f"Night events missing on {night}; run the night-event phase first.")

        # Stage 1 for every site, as the aggregator does: alt/az does not depend
        # on which site observes the target, and the scheduler may try both.
        fresh = await calc.target_data_repo.get_fresh_night_dates_for_targets(
            targets, list(sites), night, night
        )
        stage1_rows = [
            calc._stage1_row(target, sites[site_id], night_events[site_id])
            for target in targets
            for site_id in sites
            if night not in fresh.get((target.id, site_id), ())
        ]
        await calc.target_data_repo.bulk_upsert(stage1_rows)
        stage1 = {
            (row["target_id"], row["site_id"]): SimpleNamespace(**row)
            for row in stage1_rows
        }

        due = [
            request
            for name in names
            for request in _worker["requests_by_target"].get(name, ())
            if windows[request.observation_id][0] <= night <= windows[request.observation_id][1]
        ]
        stored = await calc.visibility_repo.get_stored_observation_nights(
            [r.observation_id for r in due], night, night
        )
        missing = [r for r in due if (r.observation_id, night) not in stored]

        # Only Stage 1 that was already fresh has to come back from the DB.
        to_read: dict[int, set[int]] = {}
        for request in missing:
            key = (_worker["targets"][request.target_name].id, SITE_KEY_TO_ID[request.site_id])
            if key not in stage1:
                to_read.setdefault(key[1], set()).add(key[0])
        for site_id, target_ids in to_read.items():
            for data in await calc.target_data_repo.get_for_targets_on_night(
                list(target_ids), site_id, night
            ):
                stage1[(data.target_id, data.site_id)] = data

        stage2_rows = []
        for request in missing:
            target = _worker["targets"][request.target_name]
            site_id = SITE_KEY_TO_ID[request.site_id]
            data = stage1.get((target.id, site_id))
            if data is None:
                continue
            result = calc._calculate_stage2(request, night_events[site_id], data, target.name)
            stage2_rows.append(dict(
                observation_id=result.observation_id,
                target_id=target.id,
                site_id=site_id,
                night_date=result.night_date,
                remaining_minutes=result.remaining_minutes,
                visible_ranges=result.visible_ranges,
            ))
        await calc.visibility_repo.bulk_upsert(stage2_rows)

    return ChunkResult(len(stage1_rows), len(stage2_rows))


# --- parent side ---------------------------------------------------------------

def execute(plan: Plan, workers: int) -> int:
    """Run the plan in a process pool. Returns the number of failed chunks."""
    # The program set reaches the workers through a file, not initargs: the
    # parent writes initargs into each new worker's pipe and blocks until that
    # worker has imported enough to unpickle them, which serialised start-up
    # to ~1.2s per worker.
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as program_set:
        pickle.dump((plan.targets, plan.requests_by_target, plan.windows), program_set)
    try:
        return _execute(plan, workers, program_set.name)
    finally:
        os.unlink(program_set.name)


def _execute(plan: Plan, workers: int, program_set_path: str) -> int:
    context = multiprocessing.get_context("spawn")
    stop = context.Event()
    pool = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init_worker,
        initargs=(program_set_path, stop),
    )
    try:
        if plan.missing_night_events:
            t0 = time.perf_counter()
            event_futures = [
                pool.submit(fill_night_events, site_id, night)
                for site_id, night in plan.missing_night_events
            ]
            for future in as_completed(event_futures):
                # Every chunk of a night needs its night events: stop on the first failure.
                future.result()
            _logger.info(
                f"Night events: {len(event_futures)} stored in "
                f"{time.perf_counter() - t0:.1f}s."
            )

        # Submitted first so one worker starts downloading right away.
        ephemerides = (
            pool.submit(fill_ephemerides, plan.first, plan.last)
            if plan.deferred_chunks else None
        )
        futures: dict[Future, tuple[date, tuple[str, ...]]] = {
            pool.submit(fill_chunk, night, names): (night, names)
            for night, names in plan.chunks
        }
        return _drain_chunks(pool, futures, ephemerides, plan.deferred_chunks)
    except KeyboardInterrupt:
        _logger.warning(
            "Interrupted: cancelling queued chunks and waiting for running ones. "
            "Committed chunks are kept; re-run to resume."
        )
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    finally:
        pool.shutdown(wait=True)


def _drain_chunks(
    pool: ProcessPoolExecutor,
    futures: dict[Future, tuple[date, tuple[str, ...]]],
    ephemerides: Optional[Future],
    deferred_chunks: list[tuple[date, tuple[str, ...]]],
) -> int:
    """Collect chunk results; queue the deferred chunks once ``ephemerides`` is done."""
    t0 = time.perf_counter()
    last_log = t0
    total = len(futures) + len(deferred_chunks)
    done = failed = stage1 = stage2 = 0
    pending = set(futures) | ({ephemerides} if ephemerides is not None else set())
    while pending:
        finished, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in finished:
            if future is ephemerides:
                _report_ephemerides(future)
                for night, names in deferred_chunks:
                    chunk = pool.submit(fill_chunk, night, names)
                    futures[chunk] = (night, names)
                    pending.add(chunk)
                continue
            done += 1
            try:
                result = future.result()
                stage1 += result.stage1_rows
                stage2 += result.stage2_rows
            except Exception as exc:
                failed += 1
                night, names = futures[future]
                _logger.error(
                    f"Chunk {night} ({len(names)} targets from {names[0]!r}) failed; "
                    f"it is retried on the next run: {exc!r}"
                )
        now = time.perf_counter()
        if now - last_log >= _PROGRESS_EVERY_SECONDS or not pending:
            last_log = now
            eta = progress_eta_seconds(elapsed_seconds=now - t0, done=done, total=total)
            _logger.info(
                f"Chunks {done}/{total} ({failed} failed): {stage1} Stage 1 "
                f"and {stage2} Stage 2 rows stored in {_format_duration(now - t0)}; "
                f"ETA ~{'unknown' if eta is None else _format_duration(eta)}."
            )
    return failed


def _report_ephemerides(future: Future) -> None:
    try:
        result = future.result()
    except Exception as exc:
        # The deferred chunks still run: each one fetches what it is missing.
        _logger.error(f"Horizons download failed ({exc!r}); non-sidereal chunks fetch their own files.")
        return
    _logger.info(
        f"Horizons: {result.ready} ephemeris files ready in {_format_duration(result.seconds)}"
        f"{f', {len(result.failed)} failed' if result.failed else ''}; "
        f"starting the non-sidereal chunks."
    )
    for failure in result.failed[:10]:
        _logger.warning(f"  {failure}")
    if len(result.failed) > 10:
        _logger.warning(f"  ... and {len(result.failed) - 10} more.")


async def _prepare(args: argparse.Namespace) -> Optional[Plan]:
    ObservatoryProperties.set_properties(GeminiProperties)
    labels = await gpp.client.scheduler.get_all_reference_labels()
    program_ids = [label[1] for label in labels]
    t0 = time.perf_counter()
    payloads, requests, windows, _, counts = await _collect_requests(
        program_ids, MemoryGuard(None)
    )
    _logger.info(
        f"Parsed {len(program_ids)} programs in {time.perf_counter() - t0:.1f}s: "
        f"{len(payloads)} targets, {len(requests)} observations "
        f"({counts['skipped_no_target']} without a usable base target)."
    )
    await init_db_engine()
    try:
        return await build_plan(
            payloads, requests, windows,
            start=args.start, end=args.end,
            chunk_size=args.chunk_size, dry_run=args.dry_run,
        )
    finally:
        await dispose_engine()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fill the Sight DB from the ODB program set using a process pool (local only)."
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
        help="worker processes, each with its own DB connection (default: CPUs - 2)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"targets per task (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--start", type=date.fromisoformat,
        help="first night, YYYY-MM-DD (default: earliest program window)",
    )
    parser.add_argument(
        "--end", type=date.fromisoformat,
        help="last night, YYYY-MM-DD (default: latest program window)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="parse and report the plan; write nothing",
    )
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    plan = asyncio.run(_prepare(args))
    if plan is None:
        return 0
    try:
        failed = execute(plan, args.workers)
    except KeyboardInterrupt:
        return 130
    _logger.info(
        f"Done in {_format_duration(time.perf_counter() - t0)} with {args.workers} workers"
        f"{f'; {failed} chunks failed, re-run to retry them' if failed else ''}."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
