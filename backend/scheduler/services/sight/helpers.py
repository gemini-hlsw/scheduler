"""Helpers that adapt sight's per-night outputs into scheduler/collector shapes.

The outbound counterpart to ``_temporary/lucupy_adapters`` (which adapts
lucupy -> sight): these turn sight Stage-1/Stage-2 results into the
``TargetInfo`` and visibility-fraction structures the Collector consumes. Kept
out of the Collector so it stays free of this glue.
"""

import numpy as np
from astropy.coordinates import Angle, SkyCoord
import astropy.units as u

from lucupy.minimodel import NightIndex

from scheduler.core.calculations.targetinfo import TargetInfo


def resize_to(arr: np.ndarray, expected_length: int) -> np.ndarray:
    """Pad (by repeating the last value) or truncate `arr` to `expected_length`.

    Sight's Stage-1 arrays are sized from the timestamps stored in its
    `night_events` rows, while the scheduler's `night_events.times[night_idx]`
    is recomputed fresh from astropy each run. Slight rounding/ephemeris
    differences can leave the lengths off by ±1, which downstream broadcasting
    rejects. One-slot padding/truncation is acceptable: the fence values are
    near twilight where astronomical visibility is already ~zero.
    """
    n = len(arr)
    if n == expected_length:
        return arr
    if n > expected_length:
        return arr[:expected_length]
    pad = np.full(expected_length - n, arr[-1], dtype=arr.dtype)
    return np.concatenate([arr, pad])


def cumulative_remaining_by_night(
    rem_min_by_night: dict[NightIndex, dict[str, int]],
) -> dict[NightIndex, dict[str, int]]:
    """Backward cumulative remaining minutes per observation, per night.

    Reproduces the legacy ``get_target_visibility`` denominator: the value for an
    observation on night ``n`` is the sum of its remaining minutes from night
    ``n`` through the last night in the map. Input is expected to be already
    resource/program gated (nights an observation cannot use are simply absent
    and therefore contribute 0, exactly as the old code zeroed them). Only the
    observations visible on a given night are kept in that night's entry.
    """
    cumulative_by_night: dict[NightIndex, dict[str, int]] = {}
    running: dict[str, int] = {}
    for night_index in sorted(rem_min_by_night, reverse=True):
        for obs_id, rem in rem_min_by_night[night_index].items():
            running[obs_id] = running.get(obs_id, 0) + rem
        cumulative_by_night[night_index] = {
            obs_id: running[obs_id] for obs_id in rem_min_by_night[night_index]
        }
    return cumulative_by_night


def build_target_info(stage1_entry: dict, rem_visibility_frac: float,
                      expected_length: int) -> TargetInfo:
    """Construct a TargetInfo from a Sight Stage-1 night entry.

    Stage-1 returns ra/dec in degrees and alt/az/hourangle in radians; airmass
    is unitless. Arrays are resized to `expected_length` (the scheduler's
    `len(night_events.times[night_idx])`) so downstream consumers that combine
    target arrays with night_events-sized arrays (variants, wind, conditions)
    don't crash on broadcasting mismatches.
    """
    ra = resize_to(np.asarray(stage1_entry['ra']), expected_length)
    dec = resize_to(np.asarray(stage1_entry['dec']), expected_length)
    alt = resize_to(np.asarray(stage1_entry['alt']), expected_length)
    az = resize_to(np.asarray(stage1_entry['az']), expected_length)
    hourangle = resize_to(np.asarray(stage1_entry['hourangle']), expected_length)
    airmass = resize_to(np.asarray(stage1_entry['airmass']), expected_length)

    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
    return TargetInfo(
        coord=coord,
        alt=Angle(alt, unit=u.rad),
        az=Angle(az, unit=u.rad),
        hourangle=Angle(hourangle, unit=u.rad),
        airmass=airmass,
        visibility_slot_idx=np.arange(expected_length, dtype=int),
        rem_visibility_frac=rem_visibility_frac,
    )
