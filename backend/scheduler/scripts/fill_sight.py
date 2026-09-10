#!/usr/bin/env python3
# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""Populate the in-process sight DB with the full GN+GS program set.

Loads programs through the validation path (OcsProgramProvider + bundled
programs.zip) restricted to the IDs in ``program_ids.redis.txt`` (160 GN +
116 GS = 276 unique programs) over 2018-08-01 .. 2019-01-31, then drives the
sight Calculator directly (no HTTP) to:

  1. Stage 1 — create target rows and pre-compute RA/Dec/alt/az/airmass/
     hourangle/par_ang for every (target, site, night) in the range.
  2. Stage 1 (precompute) — ensure alt/airmass arrays exist for every
     (target, site, night) the scheduler will request.
  3. Stage 2 — calculate and store per-night remaining_minutes and
     visible_ranges for each observation.

Re-running is safe: targets already in the DB are skipped (sight raises
"Target '...' already exists"), and Stage 1 / Stage 2 upserts re-fill missing
rows without touching existing ones.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from lucupy.minimodel import ObservationClass
from lucupy.minimodel.semester import Semester
from lucupy.minimodel.target import NonsiderealTarget, SiderealTarget
from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties
from lucupy.types import ZeroTime
from sqlalchemy import func as sa_func, select as sa_select

from definitions import ROOT_DIR
from scheduler.core.programprovider.ocs import OcsProgramProvider, ocs_program_data
from scheduler.core.sources.sources import Sources
from scheduler.services import logger_factory
from scheduler.services.sight.calculator.calculator import Calculator
from scheduler.services.sight.calculator.constants import SITE_KEY_TO_ID
from scheduler.services.sight.database.models import VisibilityData
from scheduler.services.sight.calculator.models import (
    ElevationType,
    ObservationConstraints,
    ObservationRequest,
    TargetCreate,
)
from scheduler.services.sight.database.connection import (
    dispose_engine,
    init_db_engine,
    session_scope,
)
from scheduler.services.sight._temporary.lucupy_adapters import (
    expand_timing_windows,
    program_window,
)


_logger = logger_factory.create_logger(__name__)


START = datetime(2018, 8, 1, 8, 0, 0, tzinfo=timezone.utc)
END = datetime(2019, 1, 31, 8, 0, 0, tzinfo=timezone.utc)

PROGRAM_IDS_PATH = Path(ROOT_DIR) / 'scheduler' / 'data' / 'program_ids.redis.txt'

# Sites to seed, processed in this order (GN fully, then GS fully). Stage 1
# is scoped per site (see create_targets_bulk's site_ids param), so a target
# only observed at one site never triggers Horizons/DB work for the other.
SITES_TO_SEED = ('GN','GS')

# Retries for a single unit of work (one target's Stage 1, one night's Stage
# 2) when its DB session dies mid-operation (e.g. a Heroku Postgres
# connection reset over a long-running WAN connection). Each attempt opens a
# brand-new session_scope(), so pool_pre_ping gets a chance to swap in a live
# connection instead of the run just crashing on the first drop.
DB_RETRY_ATTEMPTS = 3
DB_RETRY_DELAY_SECONDS = 5

# Non-sidereal Stage 1 reads ephemerides from here (HorizonsClient.path);
# a miss falls through to a live, un-timed-out requests.get() to
# ssd.jpl.nasa.gov, which is what makes a missing-cache run look "stuck".
HORIZONS_CACHE_DIR = Path(ROOT_DIR) / 'scheduler' / 'services' / 'horizons' / 'data'


def _target_payload(target) -> Optional[TargetCreate]:
    if target is None:
        return None
    if isinstance(target, SiderealTarget):
        return TargetCreate(
            name=str(target.name),
            is_sidereal=True,
            base_ra=float(target.ra),
            base_dec=float(target.dec),
            pm_ra=float(target.pm_ra) if target.pm_ra is not None else None,
            pm_dec=float(target.pm_dec) if target.pm_dec is not None else None,
            epoch=float(target.epoch) if target.epoch is not None else 2000.0,
        )
    if isinstance(target, NonsiderealTarget):
        return TargetCreate(
            name=str(target.name),
            is_sidereal=False,
            # Sight requires base_ra/base_dec; non-sidereal targets get resolved
            # via horizons_id, but the schema still has ge=0 / ge=-90 bounds, so
            # we send neutral placeholders.
            base_ra=0.0,
            base_dec=0.0,
            horizons_id=str(target.des) if target.des is not None else None,
            tag=target.tag.name.lower() if target.tag is not None else None,
        )
    return None


async def _stored_obs_count(session, obs_ids: List[str], night_date) -> int:
    """How many of ``obs_ids`` already have a visibility row for the night.

    One tiny COUNT query (no visibility rows downloaded) so re-runs skip
    fully stored nights instantly. DISTINCT guards against an observation
    having rows under two target_ids after a retarget.
    """
    stmt = sa_select(
        sa_func.count(sa_func.distinct(VisibilityData.observation_id))
    ).where(
        VisibilityData.observation_id.in_(obs_ids),
        VisibilityData.night_date == night_date,
    )
    return (await session.execute(stmt)).scalar_one()


def _constraints_payload(
    constraints, range_end: datetime, program_start=None, program_end=None
) -> ObservationConstraints:
    if constraints is None:
        return ObservationConstraints(
            timing_windows=program_window(program_start, program_end),
        )
    cond = getattr(constraints, 'conditions', None)
    target_sb = float(cond.sb.value) if (cond is not None and cond.sb is not None) else 1.0
    timing_windows = expand_timing_windows(
        getattr(constraints, 'timing_windows', None), range_end
    )
    if not timing_windows:
        timing_windows = program_window(program_start, program_end)
    return ObservationConstraints(
        target_sb=target_sb,
        elevation_type=ElevationType(constraints.elevation_type.name.lower()),
        elevation_min=float(constraints.elevation_min),
        elevation_max=float(constraints.elevation_max),
        timing_windows=timing_windows,
        has_resources=True,
        can_schedule=True,
    )


def _load_observations(semesters, program_ids_path: Path) -> list:
    """Parse the bundled OCS programs filtered by ``program_ids_path``.

    Returns a list of ``(observation, program_start, program_end)`` tuples so the
    program's active period can bound observations without explicit timing
    windows.
    """
    obs_classes = frozenset({
        ObservationClass.SCIENCE,
        ObservationClass.PROGCAL,
        ObservationClass.PARTNERCAL,
    })
    provider = OcsProgramProvider(obs_classes, Sources())

    all_obs = []
    bad_programs = 0
    for json_program in ocs_program_data(program_ids_path):
        try:
            if len(json_program.keys()) != 1:
                continue
            data = next(iter(json_program.values()))
            program = provider.parse_program(data)
            if program is None:
                continue
            if program.semester is None or program.semester not in semesters:
                continue
            if program.program_awarded() == ZeroTime:
                continue
            for obs in program.observations():
                all_obs.append((obs, program.start, program.end))
        except Exception as e:
            bad_programs += 1
            _logger.debug(f'Failed to parse program: {e}')

    if bad_programs:
        _logger.info(f'Skipped {bad_programs} unparseable programs.')
    return all_obs


def _check_horizons_cache(
    targets: List[TargetCreate],
    start_date,
    end_date,
    sites=SITES_TO_SEED,
) -> None:
    """Log which non-sidereal (site, target, night) ephemeris files are
    missing from HORIZONS_CACHE_DIR before Stage 1 runs for ``sites``.

    A cache miss falls through to a live, un-timed-out requests.get() to
    JPL Horizons -- so this is a pure filesystem check (no DB, no network)
    run up front to surface how much of that is coming, instead of the run
    just going silent on a blocking HTTP call.
    """
    nonsidereal = [t for t in targets if not t.is_sidereal and t.horizons_id]
    if not nonsidereal:
        _logger.info('Horizons cache check: no non-sidereal targets in this run.')
        return

    total_nights = (end_date - start_date).days + 1
    total_needed = 0
    missing_by_target: Dict[str, int] = {}

    for target in nonsidereal:
        targ_name = target.horizons_id.replace(' ', '_').replace('/', '')
        for site in sites:
            current_date = start_date
            while current_date <= end_date:
                total_needed += 1
                filename = f'{site}_{targ_name}_{current_date.strftime("%Y%m%d")}UT.eph'
                if not (HORIZONS_CACHE_DIR / filename).exists():
                    missing_by_target[target.name] = missing_by_target.get(target.name, 0) + 1
                current_date += timedelta(days=1)

    total_missing = sum(missing_by_target.values())
    if not total_missing:
        _logger.info(
            f'Horizons cache check: all {total_needed} required ephemeris files '
            f'for {len(nonsidereal)} non-sidereal targets are cached.'
        )
        return

    per_target_max = total_nights * len(sites)
    _logger.warning(
        f'Horizons cache check: {total_missing}/{total_needed} required ephemeris '
        f'files are MISSING for {len(missing_by_target)}/{len(nonsidereal)} non-sidereal '
        f'targets -- Stage 1 will block on live ssd.jpl.nasa.gov calls to fill these in:'
    )
    for name, count in sorted(missing_by_target.items(), key=lambda kv: -kv[1])[:20]:
        _logger.warning(f'  {name}: {count}/{per_target_max} (site x night) files missing')
    if len(missing_by_target) > 20:
        _logger.warning(f'  ... and {len(missing_by_target) - 20} more targets.')


async def _run() -> None:
    ObservatoryProperties.set_properties(GeminiProperties)

    semesters = frozenset([
        Semester.find_semester_from_date(START),
        Semester.find_semester_from_date(END),
    ])

    _logger.info(
        f'Loading observations from {PROGRAM_IDS_PATH.name}: '
        f'{START.date()} -> {END.date()}, sites={list(SITES_TO_SEED)}, '
        f'semesters={[str(s) for s in semesters]}'
    )

    obs_list = _load_observations(semesters, PROGRAM_IDS_PATH)
    if not obs_list:
        raise SystemExit('No observations were parsed from the OCS programs.')

    by_site: Dict[str, int] = {}
    for o, _ps, _pe in obs_list:
        by_site[o.site.name] = by_site.get(o.site.name, 0) + 1
    _logger.info(
        f'Loaded {len(obs_list)} observations '
        f'({", ".join(f"{k}={v}" for k, v in sorted(by_site.items()))}).'
    )

    obs_by_site: Dict[str, list] = {site: [] for site in SITES_TO_SEED}
    for obs, prog_start, prog_end in obs_list:
        obs_by_site.setdefault(obs.site.name, []).append((obs, prog_start, prog_end))

    await init_db_engine()
    try:
        start_date, end_date = START.date(), END.date()
        nights_total = (end_date - start_date).days + 1

        created_total = 0
        failed_total = 0
        stored_total = 0
        all_errors: List[str] = []

        for site in SITES_TO_SEED:
            site_obs = obs_by_site.get(site, [])

            targets_by_name: Dict[str, TargetCreate] = {}
            requests: List[ObservationRequest] = []
            skipped_no_target = 0

            for obs, prog_start, prog_end in site_obs:
                base = obs.base_target()
                if base is None:
                    skipped_no_target += 1
                    continue
                target = _target_payload(base)
                if target is None:
                    skipped_no_target += 1
                    continue
                targets_by_name.setdefault(target.name, target)
                requests.append(ObservationRequest(
                    observation_id=obs.id.id,
                    target_name=str(base.name),
                    site_id=obs.site.name,
                    constraints=_constraints_payload(
                        getattr(obs, 'constraints', None), END, prog_start, prog_end
                    ),
                ))

            site_targets = list(targets_by_name.values())
            _logger.info(
                f'{site}: {len(site_obs)} observations, {len(site_targets)} targets, '
                f'{len(requests)} requests (skipped {skipped_no_target} with no base target).'
            )
            _check_horizons_cache(site_targets, start_date, end_date, sites=[site])

            _logger.info(
                f'{site} Stage 1 — creating targets + precomputing alt/az/airmass '
                f'({len(site_targets)} targets)...'
            )
            for i, target in enumerate(site_targets, start=1):
                for attempt in range(1, DB_RETRY_ATTEMPTS + 1):
                    try:
                        async with session_scope() as session:
                            calc = Calculator(session)
                            stage1 = await calc.create_targets_bulk(
                                [target], start_date, end_date, site_ids=[site]
                            )
                            await calc.precompute_stage1(
                                start_date=start_date,
                                end_date=end_date,
                                target_names=[target.name],
                                site_ids=[site],
                            )
                        created_total += stage1.created
                        failed_total += stage1.failed
                        all_errors.extend(stage1.errors)
                        for err in stage1.errors:
                            _logger.warning(f'{site} Stage 1 target error: {err}')
                        break
                    except Exception as exc:
                        if attempt == DB_RETRY_ATTEMPTS:
                            failed_total += 1
                            msg = f"{target.name}: giving up after {DB_RETRY_ATTEMPTS} attempts: {exc}"
                            all_errors.append(msg)
                            _logger.error(f'{site} Stage 1: {msg}')
                        else:
                            _logger.warning(
                                f'{site} Stage 1: DB error on {target.name} '
                                f'(attempt {attempt}/{DB_RETRY_ATTEMPTS}), retrying with a '
                                f'fresh connection: {exc}'
                            )
                            await asyncio.sleep(DB_RETRY_DELAY_SECONDS)

                _logger.info(
                    f'{site} Stage 1: {i}/{len(site_targets)} targets done '
                    f'(created={created_total} failed={failed_total}) — {target.name}'
                )

            _logger.info(f'{site} Stage 2 — calculating and storing per-night visibility...')
            site_id_int = SITE_KEY_TO_ID[site]
            obs_ids = [r.observation_id for r in requests]
            request_names = list({r.target_name for r in requests})
            constraints_by_obs = {
                r.observation_id: r.constraints.model_dump(mode='json') for r in requests
            }
            site_stored = 0
            nights_skipped = 0
            night_index = 0
            current_date = start_date
            while current_date <= end_date:
                night_index += 1
                for attempt in range(1, DB_RETRY_ATTEMPTS + 1):
                    try:
                        night_stored = 0
                        async with session_scope() as session:
                            calc = Calculator(session)

                            stored_count = await _stored_obs_count(session, obs_ids, current_date)
                            if stored_count >= len(obs_ids):
                                nights_skipped += 1
                                break

                            db_targets = await calc.target_repo.get_by_names(request_names)
                            night_event = await calc.night_repo.get_by_site_and_night(
                                site_id_int, current_date
                            )
                            if night_event is None:
                                _logger.warning(
                                    f'{site} Stage 2: no night event for {current_date} '
                                    f'(Stage 1 incomplete?); skipping night.'
                                )
                                break

                            stage1_by_target = {
                                d.target_id: d
                                for d in await calc.target_data_repo.get_for_targets_on_night(
                                    [t.id for t in db_targets.values()],
                                    site_id_int,
                                    current_date,
                                )
                            }

                            rows = []
                            for req in requests:
                                target = db_targets.get(req.target_name)
                                if target is None:
                                    continue
                                stage1 = stage1_by_target.get(target.id)
                                if stage1 is None:
                                    continue
                                result = calc._calculate_stage2(
                                    req, night_event, stage1, target.name
                                )
                                rows.append({
                                    'observation_id': req.observation_id,
                                    'target_id': target.id,
                                    'site_id': site_id_int,
                                    'night_date': current_date,
                                    'remaining_minutes': result.remaining_minutes,
                                    'visible_ranges': result.visible_ranges,
                                    'constraints': constraints_by_obs[req.observation_id],
                                })
                            night_stored = await calc.visibility_repo.bulk_upsert(rows)
                        site_stored += night_stored
                        _logger.info(
                            f'{site} Stage 2: {current_date} stored={night_stored} '
                            f'({night_index}/{nights_total} nights)'
                        )
                        break
                    except Exception as exc:
                        if attempt == DB_RETRY_ATTEMPTS:
                            msg = (
                                f"{site} {current_date}: giving up after "
                                f"{DB_RETRY_ATTEMPTS} attempts: {exc}"
                            )
                            all_errors.append(msg)
                            _logger.error(f'{site} Stage 2: {msg}')
                        else:
                            _logger.warning(
                                f'{site} Stage 2: DB error on {current_date} '
                                f'(attempt {attempt}/{DB_RETRY_ATTEMPTS}), retrying with a '
                                f'fresh connection: {exc}'
                            )
                            await asyncio.sleep(DB_RETRY_DELAY_SECONDS)
                current_date += timedelta(days=1)
            stored_total += site_stored
            _logger.info(
                f'{site} Stage 2 done: stored={site_stored} nights={nights_total} '
                f'skipped_complete={nights_skipped}'
            )

        for err in all_errors[:10]:
            _logger.warning(f'  target error: {err}')
        _logger.info(
            f'All sites done: created={created_total} failed={failed_total} '
            f'errors={len(all_errors)} stage2_stored={stored_total}'
        )
    finally:
        await dispose_engine()


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
