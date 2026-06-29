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
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault('REDISCLOUD_URL', 'redis://mock:6379')

from lucupy.minimodel import ObservationClass
from lucupy.minimodel.semester import Semester
from lucupy.minimodel.target import NonsiderealTarget, SiderealTarget
from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties
from lucupy.types import ZeroTime

from definitions import ROOT_DIR
from scheduler.core.programprovider.ocs import OcsProgramProvider, ocs_program_data
from scheduler.core.sources.sources import Sources
from scheduler.services import logger_factory
from scheduler.services.sight.calculator.calculator import Calculator
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

PROGRAM_IDS_PATH = Path(ROOT_DIR) / 'scheduler' / 'data' / 'program_ids_gn.redis.txt'

# Sites to seed; sight stores per (target, site) so both must be precomputed.
SITES_TO_SEED = ('GN',)


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

    targets_by_name: Dict[str, TargetCreate] = {}
    requests: List[ObservationRequest] = []
    skipped_no_target = 0

    for obs, prog_start, prog_end in obs_list:
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

    targets = list(targets_by_name.values())
    _logger.info(
        f'Prepared {len(targets)} unique targets and {len(requests)} observations '
        f'(skipped {skipped_no_target} with no base target).'
    )

    await init_db_engine()
    try:
        async with session_scope() as session:
            calc = Calculator(session)

            _logger.info('Stage 1 — creating targets + precomputing alt/az/airmass...')
            stage1 = await calc.create_targets_bulk(targets, START.date(), END.date())
            _logger.info(
                f'Targets done: created={stage1.created} failed={stage1.failed} '
                f'errors={len(stage1.errors)}'
            )
            for err in stage1.errors[:10]:
                _logger.warning(f'  target error: {err}')

            target_names = [t.name for t in targets]
            _logger.info(f'Stage 1 (precompute) for sites={list(SITES_TO_SEED)}...')
            precompute = await calc.precompute_stage1(
                start_date=START.date(),
                end_date=END.date(),
                target_names=target_names,
                site_ids=list(SITES_TO_SEED),
            )
            _logger.info(
                f'Stage 1 done: targets={precompute.get("targets")} '
                f'sites={precompute.get("sites")} nights={precompute.get("nights")} '
                f'total_computations={precompute.get("total_computations")}'
            )

            _logger.info('Stage 2 — calculating and storing per-night visibility...')
            stage2 = await calc.store_visibility(requests, START.date(), END.date())
            _logger.info(
                f'Stage 2 done: stored={stage2.get("stored")} nights={stage2.get("nights")}'
            )
    finally:
        await dispose_engine()


def main() -> None:
    asyncio.run(_run())


if __name__ == '__main__':
    main()
