#!/usr/bin/env python3
# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""Populate the Sight visibility service with the full GN+GS program set.

Loads programs through the validation path (OcsProgramProvider + bundled
programs.zip) restricted to the IDs in ``program_ids.redis.txt``
(160 GN + 116 GS = 276 unique programs) over 2018-08-01 .. 2019-01-31, then
pushes:

  1. Stage 1 — POST /targets/bulk: creates target rows and pre-computes
     RA/Dec/alt/az/airmass/hourangle/par_ang for every night in the range.
  2. Stage 1 (precompute) — POST /precompute: ensures alt/airmass arrays
     exist for every (target, site, night) the scheduler will request.
  3. Stage 2 — POST /visibility/store: calculates and stores per-night
     remaining_minutes and visible_ranges for each observation.

Sight is expected to be reachable at http://localhost:9800.
"""

import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault('REDISCLOUD_URL', 'redis://mock:6379')

from lucupy.minimodel import ObservationClass, Site
from lucupy.minimodel.semester import Semester
from lucupy.minimodel.target import NonsiderealTarget, SiderealTarget
from lucupy.minimodel.timingwindow import TimingWindow
from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties
from lucupy.types import ZeroTime

from definitions import ROOT_DIR
from scheduler.clients.sight_client import SightClient
from scheduler.core.programprovider.ocs import OcsProgramProvider, ocs_program_data
from scheduler.core.sources.sources import Sources
from scheduler.services import logger_factory


_logger = logger_factory.create_logger(__name__)


SIGHT_URL = 'http://localhost:9800/api/v1'
START = datetime(2018, 8, 1, 8, 0, 0, tzinfo=timezone.utc)
END = datetime(2019, 1, 31, 8, 0, 0, tzinfo=timezone.utc)

# Consolidated GN+GS program list (matches what run.py would consume when
# pointed at program_ids.redis.txt). 160 GN + 116 GS = 276 programs.
PROGRAM_IDS_PATH = Path(ROOT_DIR) / 'scheduler' / 'data' / 'program_ids.redis.txt'

# Sites to seed; sight stores per (target, site) so both must be precomputed.
SITES_TO_SEED = ('GN', 'GS')


def _target_payload(target) -> Optional[Dict[str, Any]]:
    if target is None:
        return None
    if isinstance(target, SiderealTarget):
        return {
            'name': str(target.name),
            'is_sidereal': True,
            'base_ra': float(target.ra),
            'base_dec': float(target.dec),
            'pm_ra': float(target.pm_ra) if target.pm_ra is not None else None,
            'pm_dec': float(target.pm_dec) if target.pm_dec is not None else None,
            'epoch': float(target.epoch) if target.epoch is not None else 2000.0,
        }
    if isinstance(target, NonsiderealTarget):
        return {
            'name': str(target.name),
            'is_sidereal': False,
            # Sight requires base_ra/base_dec; non-sidereal targets get resolved
            # via horizons_id, but the schema still has ge=0 / ge=-90 bounds, so
            # we send neutral placeholders.
            'base_ra': 0.0,
            'base_dec': 0.0,
            'horizons_id': str(target.des) if target.des is not None else None,
            'tag': target.tag.name.lower() if target.tag is not None else None,
        }
    return None


def _expand_timing_windows(windows, range_end: datetime) -> List[Dict[str, str]]:
    """Expand lucupy TimingWindow repeats into flat {start, end} pairs.

    repeat == 0 (NON_REPEATING) -> one window.
    repeat  > 0                 -> repeat + 1 windows, offset by period.
    repeat == -1 (FOREVER)      -> emit until window start exceeds range_end.
    """
    out: List[Dict[str, str]] = []
    for tw in (windows or []):
        if tw.start is None or tw.duration is None:
            continue
        tw_start = tw.start.to_datetime(timezone.utc) if hasattr(tw.start, 'to_datetime') else tw.start
        if tw_start.tzinfo is None:
            tw_start = tw_start.replace(tzinfo=timezone.utc)
        if tw.repeat == TimingWindow.NON_REPEATING:
            count = 1
            period = None
        elif tw.repeat == TimingWindow.FOREVER_REPEATING:
            count = None  # unbounded; capped by range_end below
            period = tw.period
        else:
            count = tw.repeat + 1
            period = tw.period

        idx = 0
        while True:
            offset = (period * idx) if (period is not None and idx > 0) else timedelta(0)
            window_start = tw_start + offset
            if count is None and window_start > range_end:
                break
            window_end = window_start + tw.duration
            out.append({
                'start': window_start.isoformat(),
                'end': window_end.isoformat(),
            })
            idx += 1
            if count is not None and idx >= count:
                break
            if period is None:
                break
    return out


def _constraints_payload(constraints, range_end: datetime) -> Optional[Dict[str, Any]]:
    if constraints is None:
        return None
    cond = getattr(constraints, 'conditions', None)
    target_sb = float(cond.sb.value) if (cond is not None and cond.sb is not None) else 1.0
    return {
        'target_sb': target_sb,
        'elevation_type': constraints.elevation_type.name.lower(),
        'elevation_min': float(constraints.elevation_min),
        'elevation_max': float(constraints.elevation_max),
        'timing_windows': _expand_timing_windows(
            getattr(constraints, 'timing_windows', None), range_end
        ),
        'has_resources': True,
        'can_schedule': True,
    }


def _load_observations(semesters, program_ids_path: Path) -> list:
    """Parse the bundled OCS programs filtered by ``program_ids_path``.

    Skips the full Collector/Builder pipeline so we don't trigger visibility
    calculations, night events, etc. Mirrors the parse step from
    Collector.load_programs but stops at the observation list. Returns
    observations from BOTH sites; site filtering is no longer applied here
    because sight stores per (target, site) and we need to seed everything
    the scheduler may request.
    """
    obs_classes = frozenset({
        ObservationClass.SCIENCE,
        ObservationClass.PROGCAL,
        ObservationClass.PARTNERCAL,
    })
    provider = OcsProgramProvider(obs_classes, Sources())

    all_obs = []
    bad_programs = 0
    # ocs_program_data accepts a Path and uses it as the program-id allowlist
    # (lines starting with '#' are skipped automatically).
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
                all_obs.append(obs)
        except Exception as e:
            bad_programs += 1
            _logger.debug(f'Failed to parse program: {e}')

    if bad_programs:
        _logger.info(f'Skipped {bad_programs} unparseable programs.')
    return all_obs


def main() -> None:
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
    for o in obs_list:
        by_site[o.site.name] = by_site.get(o.site.name, 0) + 1
    _logger.info(
        f'Loaded {len(obs_list)} observations '
        f'({", ".join(f"{k}={v}" for k, v in sorted(by_site.items()))}).'
    )

    targets_by_name: Dict[str, Dict[str, Any]] = {}
    observations_payload: List[Dict[str, Any]] = []
    skipped_no_target = 0

    for obs in obs_list:
        base = obs.base_target()
        if base is None:
            skipped_no_target += 1
            continue
        target = _target_payload(base)
        if target is None:
            skipped_no_target += 1
            continue
        targets_by_name.setdefault(target['name'], target)

        entry: Dict[str, Any] = {
            'observation_id': obs.id.id,
            'target_name': str(base.name),
            'site_id': obs.site.name,
        }
        cpayload = _constraints_payload(getattr(obs, 'constraints', None), END)
        if cpayload is not None:
            entry['constraints'] = cpayload
        observations_payload.append(entry)

    targets_payload = list(targets_by_name.values())
    _logger.info(
        f'Prepared {len(targets_payload)} unique targets and '
        f'{len(observations_payload)} observations '
        f'(skipped {skipped_no_target} with no base target).'
    )

    sight = SightClient(api_url=SIGHT_URL)

    _logger.info(f'POST {SIGHT_URL}/targets/bulk (create targets)...')
    stage1 = sight.create_targets_bulk(targets_payload, START.date(), END.date())
    _logger.info(
        f"Targets done: created={stage1.get('created')} "
        f"failed={stage1.get('failed')} errors={len(stage1.get('errors') or [])}"
    )
    for err in (stage1.get('errors') or [])[:10]:
        _logger.warning(f'  target error: {err}')

    target_names = [t['name'] for t in targets_payload]
    _logger.info(f'POST {SIGHT_URL}/precompute (Stage 1) for sites={list(SITES_TO_SEED)}...')
    precompute = sight.precompute_stage1(
        start_date=START.date(),
        end_date=END.date(),
        target_names=target_names,
        site_ids=list(SITES_TO_SEED),
    )
    _logger.info(
        f"Stage 1 done: targets={precompute.get('targets')} "
        f"sites={precompute.get('sites')} nights={precompute.get('nights')} "
        f"total_computations={precompute.get('total_computations')}"
    )

    _logger.info(f'POST {SIGHT_URL}/visibility/store (Stage 2)...')
    stage2 = sight.store_visibility(observations_payload, START.date(), END.date())
    _logger.info(
        f"Stage 2 done: stored={stage2.get('stored')} nights={stage2.get('nights')}"
    )


if __name__ == '__main__':
    main()
