#!/usr/bin/env python3
# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""Dump the locally-computed TargetInfo for one randomly-picked GN observation
so it can be diffed against the output of the Sight visibility service.

Loads programs through the validation path (OcsProgramProvider + bundled
programs.zip), restricts the collector to Site.GN over 2018-08-01 .. 2019-01-31,
picks one GN observation at random (or via --obs-id), and writes the full
TargetInfo dump to JSON. A short fingerprint table is printed to stdout.
"""

import argparse
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault('REDISCLOUD_URL', 'redis://mock:6379')

import numpy as np
from astropy.time import Time
from lucupy.minimodel import NightIndex, ObservationID, Site
from lucupy.minimodel.semester import Semester
from lucupy.minimodel.target import NonsiderealTarget, SiderealTarget
from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties

from scheduler.core.builder.blueprint import Blueprints
from scheduler.core.builder.validationbuilder import ValidationBuilder
from scheduler.core.events.queue import EventQueue
from scheduler.core.sources.sources import Sources
from scheduler.services import logger_factory

_logger = logger_factory.create_logger(__name__)


START = datetime(2018, 8, 1, 8, 0, 0, tzinfo=timezone.utc)
END = datetime(2019, 1, 31, 8, 0, 0, tzinfo=timezone.utc)


def _array_fingerprint(arr: np.ndarray, decimals: int = 6) -> Dict[str, Any]:
    a = np.asarray(arr, dtype=float).ravel()
    if a.size == 0:
        return {'length': 0, 'min': float('nan'), 'max': float('nan'),
                'mean': float('nan'), 'std': float('nan'), 'sha256_16': ''}
    rounded = np.round(a, decimals=decimals)
    digest = hashlib.sha256(np.ascontiguousarray(rounded).tobytes()).hexdigest()[:16]
    return {
        'length': int(a.size),
        'min': float(np.nanmin(a)),
        'max': float(np.nanmax(a)),
        'mean': float(np.nanmean(a)),
        'std': float(np.nanstd(a)),
        'sha256_16': digest,
    }


def _int_array_fingerprint(arr: np.ndarray) -> Dict[str, Any]:
    a = np.asarray(arr).ravel()
    if a.size == 0:
        return {'length': 0, 'min': 0, 'max': 0, 'sha256_16': ''}
    digest = hashlib.sha256(np.ascontiguousarray(a.astype(np.int64)).tobytes()).hexdigest()[:16]
    return {
        'length': int(a.size),
        'min': int(a.min()),
        'max': int(a.max()),
        'sha256_16': digest,
    }


def _print_fingerprints(obs_id: str, target_name: str, ti_map) -> None:
    print(f'\nFingerprints for {obs_id} (target={target_name})')
    print(f'{"night":>5} {"field":<20} {"len":>6} {"min":>12} {"max":>12} {"mean":>12} {"std":>12} {"sha":>18}')
    for night_idx in sorted(ti_map.keys()):
        ti = ti_map[night_idx]
        for field, arr in (
            ('alt_deg', ti.alt.degree),
            ('az_deg', ti.az.degree),
            ('hourangle_deg', ti.hourangle.degree),
            ('airmass', ti.airmass),
        ):
            f = _array_fingerprint(arr)
            print(f'{int(night_idx):>5} {field:<20} {f["length"]:>6} '
                  f'{f["min"]:>12.6f} {f["max"]:>12.6f} {f["mean"]:>12.6f} {f["std"]:>12.6f} '
                  f'{f["sha256_16"]:>18}')
        v = _int_array_fingerprint(ti.visibility_slot_idx)
        print(f'{int(night_idx):>5} {"visibility_slot_idx":<20} {v["length"]:>6} '
              f'{(v["min"] if v["min"] is not None else 0):>12} '
              f'{(v["max"] if v["max"] is not None else 0):>12} '
              f'{"":>12} {"":>12} {(v["sha256_16"] or ""):>18}')
        print(f'{int(night_idx):>5} {"rem_visibility_frac":<20} '
              f'{"-":>6} {ti.rem_visibility_frac:>12.6f} {"":>12} {"":>12} {"":>12} {"":>18}')


def _target_to_dict(target) -> Dict[str, Any]:
    if target is None:
        return {}
    out: Dict[str, Any] = {
        'name': str(target.name),
        'type': target.type.name if hasattr(target.type, 'name') else str(target.type),
        'magnitudes': [
            {'band': m.band.name if hasattr(m.band, 'name') else str(m.band),
             'value': float(m.value),
             'error': (float(m.error) if getattr(m, 'error', None) is not None else None)}
            for m in (target.magnitudes or [])
        ],
    }
    if isinstance(target, SiderealTarget):
        out['kind'] = 'sidereal'
        out['ra_deg'] = float(target.ra)
        out['dec_deg'] = float(target.dec)
        out['pm_ra_mas_yr'] = float(target.pm_ra)
        out['pm_dec_mas_yr'] = float(target.pm_dec)
        out['epoch'] = float(target.epoch)
    elif isinstance(target, NonsiderealTarget):
        out['kind'] = 'nonsidereal'
        out['des'] = str(target.des)
        out['tag'] = target.tag.name if hasattr(target.tag, 'name') else str(target.tag)
    return out


def _constraints_to_dict(c) -> Dict[str, Any]:
    if c is None:
        return {}
    cond = getattr(c, 'conditions', None)
    cond_d = None
    if cond is not None:
        def _enum(v):
            return v.name if hasattr(v, 'name') else str(v)
        cond_d = {
            'cc': _enum(cond.cc),
            'iq': _enum(cond.iq),
            'sb': _enum(cond.sb),
            'wv': _enum(cond.wv),
        }
    timing_windows = []
    for tw in (getattr(c, 'timing_windows', None) or []):
        timing_windows.append({
            'start': tw.start.isoformat() if tw.start is not None else None,
            'duration_seconds': (tw.duration.total_seconds() if tw.duration is not None else None),
            'repeat': int(tw.repeat),
            'period_seconds': (tw.period.total_seconds() if tw.period is not None else None),
        })
    return {
        'conditions': cond_d,
        'elevation_type': c.elevation_type.name if hasattr(c.elevation_type, 'name') else str(c.elevation_type),
        'elevation_min': float(c.elevation_min),
        'elevation_max': float(c.elevation_max),
        'timing_windows': timing_windows,
        'strehl': (c.strehl.name if getattr(c, 'strehl', None) is not None and hasattr(c.strehl, 'name')
                   else (str(c.strehl) if getattr(c, 'strehl', None) is not None else None)),
    }


def _observation_to_dict(obs) -> Dict[str, Any]:
    base = obs.base_target()
    return {
        'id': obs.id.id,
        'internal_id': str(obs.internal_id),
        'program_id': obs.belongs_to.id if obs.belongs_to is not None else None,
        'order': int(obs.order) if obs.order is not None else None,
        'title': obs.title,
        'site': obs.site.name,
        'status': obs.status.name if hasattr(obs.status, 'name') else str(obs.status),
        'active': bool(obs.active),
        'priority': obs.priority.name if hasattr(obs.priority, 'name') else str(obs.priority),
        'setuptime_type': (obs.setuptime_type.name if hasattr(obs.setuptime_type, 'name')
                           else str(obs.setuptime_type)),
        'acq_overhead_seconds': obs.acq_overhead.total_seconds() if obs.acq_overhead is not None else None,
        'obs_class': obs.obs_class.name if hasattr(obs.obs_class, 'name') else str(obs.obs_class),
        'too_type': (obs.too_type.name if obs.too_type is not None and hasattr(obs.too_type, 'name')
                     else (str(obs.too_type) if obs.too_type is not None else None)),
        'preimaging': bool(obs.preimaging),
        'band': (obs.band.name if obs.band is not None and hasattr(obs.band, 'name')
                 else (str(obs.band) if obs.band is not None else None)),
        'exec_time_seconds': obs.exec_time().total_seconds(),
        'total_used_seconds': obs.total_used().total_seconds(),
        'base_target': _target_to_dict(base),
        'all_targets': [_target_to_dict(t) for t in (obs.targets or [])],
        'constraints': _constraints_to_dict(getattr(obs, 'constraints', None)),
    }


def _ti_to_dict(ti) -> Dict[str, Any]:
    return {
        'coord': {
            'ra_deg': float(ti.coord.ra.degree),
            'dec_deg': float(ti.coord.dec.degree),
        },
        'alt_deg': np.asarray(ti.alt.degree, dtype=float).tolist(),
        'az_deg': np.asarray(ti.az.degree, dtype=float).tolist(),
        'hourangle_deg': np.asarray(ti.hourangle.degree, dtype=float).tolist(),
        'airmass': np.asarray(ti.airmass, dtype=float).tolist(),
        'visibility_slot_idx': np.asarray(ti.visibility_slot_idx, dtype=int).tolist(),
        'rem_visibility_frac': float(ti.rem_visibility_frac),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42, help='RNG seed for picking the observation (default: 42)')
    parser.add_argument('--obs-id', type=str, default=None,
                        help='Specific GN observation ID to dump (overrides random pick).')
    parser.add_argument('--out', type=Path, default=None,
                        help='Output JSON path (default: ./targetinfo_<obs_id>.json)')
    args = parser.parse_args()

    ObservatoryProperties.set_properties(GeminiProperties)

    sites = frozenset([Site.GN])
    semesters = frozenset([
        Semester.find_semester_from_date(START),
        Semester.find_semester_from_date(END),
    ])
    num_nights = (END - START).days
    night_indices = frozenset(NightIndex(i) for i in range(num_nights))

    _logger.info(f'Validation collector: {START.date()} -> {END.date()}, '
                 f'num_nights={num_nights}, sites={[s.name for s in sites]}, '
                 f'semesters={[str(s) for s in semesters]}')

    queue = EventQueue(night_indices, sites)
    builder = ValidationBuilder(Sources(), queue)

    collector = builder.build_collector(
        start=START,
        end=END,
        num_of_nights=num_nights,
        sites=sites,
        semesters=semesters,
        blueprint=Blueprints.collector,
        night_times={Site.GN: (None, None)},
        program_list=None,
    )

    gn_obs = [o for o in collector.get_all_observations() if o.site == Site.GN]
    if not gn_obs:
        raise SystemExit('No GN observations were loaded by the collector.')

    if args.obs_id is not None:
        match = [o for o in gn_obs if o.id.id == args.obs_id]
        if not match:
            raise SystemExit(f'Observation {args.obs_id} not found among {len(gn_obs)} GN observations.')
        pick = match[0]
    else:
        rng = random.Random(args.seed)
        pick = rng.choice(gn_obs)

    base = pick.base_target()
    target_name = base.name if base is not None else ''
    print(f'Picked observation: {pick.id.id}  site={pick.site.name}  target={target_name}')
    print(f'Seed (used only when --obs-id absent): {args.seed}')
    print(f'GN observations available: {len(gn_obs)}')

    obs_dump = _observation_to_dict(pick)
    print('\n--- Observation input (feed this to Sight) ---')
    print(f'  id              : {obs_dump["id"]}')
    print(f'  program_id      : {obs_dump["program_id"]}')
    print(f'  site            : {obs_dump["site"]}')
    print(f'  obs_class       : {obs_dump["obs_class"]}')
    print(f'  status          : {obs_dump["status"]}  active={obs_dump["active"]}  band={obs_dump["band"]}')
    print(f'  too_type        : {obs_dump["too_type"]}  preimaging={obs_dump["preimaging"]}')
    print(f'  exec_time (s)   : {obs_dump["exec_time_seconds"]}  used (s) : {obs_dump["total_used_seconds"]}')
    bt = obs_dump['base_target']
    if bt:
        if bt.get('kind') == 'sidereal':
            print(f'  base target     : {bt["name"]} (sidereal)')
            print(f'    ra/dec (deg)  : {bt["ra_deg"]:.6f} / {bt["dec_deg"]:.6f}  epoch={bt["epoch"]}')
            print(f'    pm (mas/yr)   : ra={bt["pm_ra_mas_yr"]} dec={bt["pm_dec_mas_yr"]}')
        else:
            print(f'  base target     : {bt["name"]} (nonsidereal)  des={bt.get("des")}  tag={bt.get("tag")}')
    c = obs_dump['constraints']
    if c:
        cond = c.get('conditions') or {}
        print(f'  conditions      : cc={cond.get("cc")} iq={cond.get("iq")} '
              f'sb={cond.get("sb")} wv={cond.get("wv")}')
        print(f'  elevation       : type={c["elevation_type"]} '
              f'min={c["elevation_min"]} max={c["elevation_max"]}')
        print(f'  timing_windows  : {len(c["timing_windows"])}')
        for i, tw in enumerate(c['timing_windows']):
            print(f'    [{i}] start={tw["start"]} duration_s={tw["duration_seconds"]} '
                  f'repeat={tw["repeat"]} period_s={tw["period_seconds"]}')

    ti_map = collector.get_target_info(pick.id)
    if not ti_map:
        raise SystemExit(f'No TargetInfo computed for {pick.id.id}.')

    _print_fingerprints(pick.id.id, target_name, ti_map)

    time_grid = collector.time_grid
    nights_dump: Dict[str, Any] = {}
    for night_idx in sorted(ti_map.keys()):
        t = Time(time_grid[int(night_idx)])
        nights_dump[str(int(night_idx))] = {
            'jd': float(t.jd),
            'date_utc': t.utc.iso[:10],
            **_ti_to_dict(ti_map[night_idx]),
        }

    dump = {
        'obs_id': pick.id.id,
        'target_name': target_name,
        'site': pick.site.name,
        'time_slot_length_min': float(collector.time_slot_length.to_value('min')),
        'num_nights': num_nights,
        'start_utc': START.isoformat(),
        'end_utc': END.isoformat(),
        'seed': args.seed,
        'observation': _observation_to_dict(pick),
        'nights': nights_dump,
    }

    out_path = args.out or Path.cwd() / f'targetinfo_{pick.id.id}.json'
    with out_path.open('w') as f:
        json.dump(dump, f, indent=2, sort_keys=True)
    print(f'\nFull TargetInfo dump written to: {out_path}')


if __name__ == '__main__':
    main()
