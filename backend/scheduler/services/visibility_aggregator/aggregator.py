# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from datetime import date, datetime, timedelta, timezone
from typing import Awaitable, Callable, Optional

import numpy as np
from astropy.time import Time
from lucupy import sky
from lucupy.minimodel import ALL_SITES, ObservationClass
from lucupy.minimodel.semester import Semester
from lucupy.types import ZeroTime
from sqlalchemy.ext.asyncio import AsyncSession

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
    async for raw in program_data:
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
    stored = 0
    current = start_date
    while current <= end_date:
        existing = await calc.visibility_repo.get_by_observation_ids_on_night(
            obs_ids, current
        )
        existing_ids = {d.observation_id for d in existing}
        missing = [r for r in requests if r.observation_id not in existing_ids]

        if missing:
            result = await calc.store_visibility(missing, current, current)
            stored += int(result.get("stored", 0))
            await calc.session.commit()

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

    targets_by_name, requests, counts = await _collect_requests(
        program_ids, range_end
    )
    _logger.info(
        f"Prepared {len(targets_by_name)} sidereal targets and {len(requests)} "
        f"observations (skipped {counts['skipped_nonsidereal']} non-sidereal, "
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

    # Stage 1: create new targets (sight skips ones that already exist) and
    # pre-compute their position arrays for the range.
    created = await calc.create_targets_bulk(targets, start_date, end_date)
    await session.commit()
    _logger.info(
        f"Stage 1 targets: created={created.created} skipped/failed={created.failed}."
    )
    if heartbeat is not None:
        await heartbeat({"phase": "stage1_targets", "created": created.created})

    # Stage 1: ensure position arrays exist for every night in the range, also
    # filling gaps for targets that already existed (e.g. new semester nights).
    precompute = await calc.precompute_stage1(
        start_date, end_date, target_names=list(targets_by_name.keys())
    )
    await session.commit()
    _logger.info(
        f"Stage 1 precompute: nights={precompute.get('nights')} "
        f"computations={precompute.get('total_computations')}."
    )
    if heartbeat is not None:
        await heartbeat({"phase": "stage1_precompute", **precompute})

    # Stage 2: store visibility for observations missing it on each night.
    stored = await _store_missing_visibility(
        calc, requests, start_date, end_date, heartbeat
    )
    _logger.info(f"Stage 2: stored {stored} new visibility rows.")

    return {
        "semester": str(semester),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "programs": len(program_ids),
        "targets": len(targets),
        "targets_created": created.created,
        "observations": len(requests),
        "stage2_stored": stored,
        **counts,
    }
