# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
import time
from datetime import date, datetime, timedelta, timezone
from typing import Awaitable, Callable, Optional

import numpy as np
from astropy.time import Time
from lucupy import sky
from lucupy.minimodel import ALL_SITES, ObservationClass
from lucupy.minimodel.semester import Semester
from lucupy.types import ZeroTime
from sqlalchemy.ext.asyncio import AsyncSession

from scheduler.config import config
from scheduler.clients.gpp import gpp
from scheduler.core.programprovider.gpp import GppProgramProvider, gpp_program_data
from scheduler.core.sources.sources import Sources
from scheduler.services import logger_factory
from scheduler.services.sight._temporary.lucupy_adapters import (
    stage2_constraints,
    target_create,
)
from scheduler.services.sight.calculator.calculator import Calculator
from scheduler.services.sight.calculator.models import ObservationRequest

_logger = logger_factory.create_logger(__name__)

__all__ = ["run_aggregation", "is_night_in_progress"]

# Heartbeat callback: an async fn the runner passes so it can refresh its
# coordination row (and surface progress) between chunks of work.
Heartbeat = Callable[[dict], Awaitable[None]]

_OBS_CLASSES = frozenset({
    ObservationClass.SCIENCE,
    ObservationClass.PROGCAL,
    ObservationClass.PARTNERCAL,
})


def _as_utc_datetime(t: Time) -> datetime:
    """Astropy Time to a UTC datetime."""
    return np.atleast_1d(t.to_datetime(timezone=timezone.utc))[0]


def _format_duration(seconds: float) -> str:
    """Human-readable duration for ETA logging."""
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def is_night_in_progress(now: Optional[datetime] = None) -> bool:
    """True if it is currently dark time (12 deg twilight) at GN or GS.

    We check both the night anchored to ``now`` and to ``now - 1 day`` so a night whose twilights
    are dated to the previous calendar day is still detected.
    """
    now = now or datetime.now(timezone.utc)
    for site in ALL_SITES:
        for day_offset in (0, -1):
            try:
                events = sky.night_events(
                    Time(now + timedelta(days=day_offset)),
                    site.location,
                    site.timezone,
                )
                eve_twi = _as_utc_datetime(events[3])
                morn_twi = _as_utc_datetime(events[4])
            except Exception as exc:
                _logger.warning(f"night_events failed for {site.name}: {exc}")
                continue
            if eve_twi <= now <= morn_twi:
                _logger.info(
                    f"Dark time in progress at {site.name} "
                    f"({eve_twi.isoformat()} .. {morn_twi.isoformat()})."
                )
                return True
    return False


async def _collect_requests(program_ids: list[str], range_end: datetime):
    """Parse the available GPP programs into sidereal targets + obs requests."""
    provider = GppProgramProvider(_OBS_CLASSES, Sources())

    targets_by_name: dict = {}
    requests: list[ObservationRequest] = []
    skipped_no_target = 0
    skipped_nonsidereal = 0
    bad_programs = 0

    program_data = await gpp_program_data(program_ids)
    processed = 0
    async for raw in program_data:
        processed += 1
        # Parsing is CPU-bound and the async-generator body has no awaits, so a
        # large semester-start dump would otherwise block the event loop (and
        # starve the runner's heartbeat task). Yield periodically so heartbeats
        # keep firing and the coordination row stays fresh.
        if processed % 25 == 0:
            await asyncio.sleep(0)
        try:
            data = next(iter(raw.values())) if len(raw.keys()) == 1 else raw
            program = provider.parse_program(data)
            if program is None:
                continue
            if program.program_awarded() == ZeroTime:
                continue

            for obs in program.observations():
                base = obs.base_target()
                # We are skipping no target observations as they should not be ready
                if base is None:
                    skipped_no_target += 1
                    continue
                payload = target_create(base)
                if payload is None:
                    skipped_no_target += 1
                    continue
                # TODO: Add NonSideral support
                if not payload.is_sidereal:
                    skipped_nonsidereal += 1
                    continue

                targets_by_name.setdefault(payload.name, payload)
                requests.append(ObservationRequest(
                    observation_id=obs.id.id,
                    target_name=str(base.name),
                    site_id=obs.site.name,
                    constraints=stage2_constraints(
                        obs,
                        has_resources=True,
                        can_schedule=True,
                        range_end=range_end,
                        program_start=program.start,
                        program_end=program.end,
                    ),
                ))
        except Exception as exc:
            bad_programs += 1
            _logger.warning(f"Failed to process a program: {exc}")

    if bad_programs:
        _logger.info(f"Skipped {bad_programs} unparseable programs.")
    counts = {
        "skipped_no_target": skipped_no_target,
        "skipped_nonsidereal": skipped_nonsidereal,
    }
    return targets_by_name, requests, counts


async def _store_missing_visibility(
    calc: Calculator,
    requests: list[ObservationRequest],
    start_date: date,
    end_date: date,
    heartbeat: Optional[Heartbeat],
) -> int:
    """Store Stage 2 only for (observation, night) pairs not already present.

    This is the "add only the ones not there" step: without it,
    ``store_visibility`` would recompute the whole semester on every cron tick.
    Commits per night to bound transaction size and persist progress.
    """
    if not requests:
        return 0

    obs_ids = [r.observation_id for r in requests]
    total_nights = (end_date - start_date).days + 1
    loop_t0 = time.perf_counter()
    stored = 0
    current = start_date
    while current <= end_date:
        existing = await calc.visibility_repo.get_by_observation_ids_on_night(
            obs_ids, current
        )
        existing_ids = {d.observation_id for d in existing}
        missing = [r for r in requests if r.observation_id not in existing_ids]

        if missing:
            t0 = time.perf_counter()
            result = await calc.store_visibility(missing, current, current)
            await calc.session.commit()
            elapsed = time.perf_counter() - t0
            night_stored = int(result.get("stored", 0))
            stored += night_stored

            # Moving ETA: measured seconds/night so far x nights remaining.
            nights_done = (current - start_date).days + 1
            eta = (time.perf_counter() - loop_t0) / nights_done * (
                total_nights - nights_done
            )
            _logger.info(
                f"Stage 2 {current.isoformat()} [{nights_done}/{total_nights}]: "
                f"{len(missing)} missing obs, stored {night_stored} in {elapsed:.1f}s "
                f"({elapsed / len(missing) * 1000:.0f} ms/obs); "
                f"ETA ~{_format_duration(eta)}."
            )

        if heartbeat is not None:
            await heartbeat({
                "phase": "stage2",
                "night": current.isoformat(),
                "stored": stored,
            })
        current += timedelta(days=1)

    return stored


async def run_aggregation(
    session: AsyncSession,
    *,
    heartbeat: Optional[Heartbeat] = None,
) -> dict:
    """Bring the Sight DB up to date for the current semester (sidereal only).

    session (AsyncSession): is the compute session (its own connection).
    heartbeat (Optional[Heartbeat]): is an optional async callback used to keep the coordination row fresh.
    """
    run_t0 = time.perf_counter()
    today = datetime.now(timezone.utc).date()
    semester = Semester.find_semester_from_date(today)
    start_date = semester.start_date()
    end_date = semester.end_date()
    range_end = datetime.combine(end_date, datetime.min.time(), tzinfo=timezone.utc)

    labels = await gpp.client.scheduler.get_all_reference_labels()
    program_ids = [label[1] for label in labels]
    _logger.info(
        f"{len(program_ids)} available programs; semester {semester} "
        f"({start_date} .. {end_date})."
    )

    parse_t0 = time.perf_counter()
    targets_by_name, requests, counts = await _collect_requests(
        program_ids, range_end
    )
    parse_elapsed = time.perf_counter() - parse_t0
    _logger.info(
        f"Prepared {len(targets_by_name)} sidereal targets and {len(requests)} "
        f"observations in {parse_elapsed:.1f}s (skipped "
        f"{counts['skipped_nonsidereal']} non-sidereal, "
        f"{counts['skipped_no_target']} without a usable base target)."
    )
    if heartbeat is not None:
        await heartbeat({
            "phase": "parsed",
            "programs": len(program_ids),
            "targets": len(targets_by_name),
            "observations": len(requests),
        })

    calc = Calculator(session)
    targets = list(targets_by_name.values())
    batch_size = max(1, int(config.visibility_aggregator.target_batch_size))

    # Stage 1, in batches: create new targets (sight skips existing ones) and
    # ensure position arrays exist for every night in the range — including
    # gaps for targets that already existed (e.g. new semester nights).
    #
    # We commit per batch so a long semester-start backfill never holds one
    # giant transaction, and so progress persists.
    # A dyno killed mid-run just resumes from the remaining gaps on the next cron tick (every upsert is idempotent).
    created_total = 0
    stage1_t0 = time.perf_counter()
    for offset in range(0, len(targets), batch_size):
        chunk = targets[offset:offset + batch_size]
        batch_t0 = time.perf_counter()
        created = await calc.create_targets_bulk(chunk, start_date, end_date)
        created_total += created.created
        await calc.precompute_stage1(
            start_date, end_date, target_names=[t.name for t in chunk]
        )
        await session.commit()
        batch_elapsed = time.perf_counter() - batch_t0
        done = min(offset + batch_size, len(targets))
        _logger.info(
            f"Stage 1 batch {offset // batch_size + 1}: {len(chunk)} targets in "
            f"{batch_elapsed:.1f}s ({batch_elapsed / len(chunk):.2f}s/target); "
            f"{done}/{len(targets)} done."
        )
        if heartbeat is not None:
            await heartbeat({
                "phase": "stage1",
                "targets_done": done,
                "targets_total": len(targets),
                "created": created_total,
            })
    stage1_elapsed = time.perf_counter() - stage1_t0
    avg_target = stage1_elapsed / len(targets) if targets else 0.0
    _logger.info(
        f"Stage 1 done: {created_total} new targets; {len(targets)} targets "
        f"ensured over {start_date}..{end_date} in {stage1_elapsed:.1f}s "
        f"({avg_target:.2f}s/target avg)."
    )

    # Stage 2: store visibility for observations missing it on each night.
    stage2_t0 = time.perf_counter()
    stored = await _store_missing_visibility(
        calc, requests, start_date, end_date, heartbeat
    )
    stage2_elapsed = time.perf_counter() - stage2_t0
    _logger.info(f"Stage 2: stored {stored} new visibility rows in {stage2_elapsed:.1f}s.")

    total_elapsed = time.perf_counter() - run_t0
    _logger.info(
        f"Aggregation timing: total={total_elapsed:.1f}s "
        f"(parse={parse_elapsed:.1f}s, stage1={stage1_elapsed:.1f}s, "
        f"stage2={stage2_elapsed:.1f}s); {len(targets)} targets, "
        f"{stored} stage-2 rows inserted."
    )

    return {
        "semester": str(semester),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "programs": len(program_ids),
        "targets": len(targets),
        "targets_created": created_total,
        "observations": len(requests),
        "stage2_stored": stored,
        "elapsed_seconds": round(total_elapsed, 1),
        "parse_seconds": round(parse_elapsed, 1),
        "stage1_seconds": round(stage1_elapsed, 1),
        "stage2_seconds": round(stage2_elapsed, 1),
        **counts,
    }
