# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

"""The sight-path visibility-fraction denominator must reproduce the legacy
``get_target_visibility`` exactly: a *per-night backward* cumulative sum of the
(resource/program gated) remaining minutes from each night through the end of
the period.
"""

from lucupy.minimodel import NightIndex

from scheduler.services.sight.helpers import cumulative_remaining_by_night


def _old_reference(per_night_minutes, num_nights, time_slot_seconds=60):
    """Faithful re-implementation of the legacy get_target_visibility denominator.

    ``per_night_minutes`` maps obs_id -> {night_index: visible_minutes}. The old
    code iterated the whole period in reverse, accumulating visibility_time
    (seconds) and dividing the (fixed) numerator by it. Here we return the
    accumulated *minutes* per (obs, night) so it can be compared directly with
    the helper's output.
    """
    out = {}
    for obs_id, by_night in per_night_minutes.items():
        running_s = 0.0
        for night_idx in reversed(range(num_nights)):
            running_s += by_night.get(night_idx, 0) * time_slot_seconds
            out.setdefault(NightIndex(night_idx), {})[obs_id] = running_s / time_slot_seconds
    return out


def _to_rem_min_by_night(per_night_minutes, num_nights):
    """Invert obs->night->minutes into the night->obs->minutes the helper takes.

    A night absent for an obs (e.g. no resources) is simply not added — exactly
    how the collector never queries those (obs, night) pairs.
    """
    rem_min_by_night = {NightIndex(n): {} for n in range(num_nights)}
    for obs_id, by_night in per_night_minutes.items():
        for night_idx, minutes in by_night.items():
            rem_min_by_night[NightIndex(night_idx)][obs_id] = minutes
    return rem_min_by_night


def test_backward_cumulative_matches_old_reference():
    num_nights = 5
    # o1 visible every night; o2 missing nights 1 and 3 (e.g. no resources).
    per_night_minutes = {
        'o1': {0: 100, 1: 80, 2: 60, 3: 40, 4: 20},
        'o2': {0: 50, 2: 30, 4: 10},
    }
    rem_min_by_night = _to_rem_min_by_night(per_night_minutes, num_nights)

    got = cumulative_remaining_by_night(rem_min_by_night)
    expected = _old_reference(per_night_minutes, num_nights)

    # Compare only the (obs, night) pairs the obs is actually visible: those are
    # the entries the collector builds TargetInfo for.
    for night_idx, by_night in rem_min_by_night.items():
        for obs_id in by_night:
            assert got[night_idx][obs_id] == expected[night_idx][obs_id]


def test_denominator_shrinks_toward_end_of_period():
    num_nights = 3
    rem_min_by_night = _to_rem_min_by_night({'o1': {0: 30, 1: 20, 2: 10}}, num_nights)
    got = cumulative_remaining_by_night(rem_min_by_night)
    # night0 sees all 3 nights, night1 sees nights 1-2, night2 only itself.
    assert got[NightIndex(0)]['o1'] == 60
    assert got[NightIndex(1)]['o1'] == 30
    assert got[NightIndex(2)]['o1'] == 10


def test_missing_nights_contribute_zero():
    num_nights = 4
    # o1 only visible on night 0; nights 1-3 absent (no resources).
    rem_min_by_night = _to_rem_min_by_night({'o1': {0: 45}}, num_nights)
    got = cumulative_remaining_by_night(rem_min_by_night)
    # Only night 0 has an entry, equal to its own minutes (later nights are 0).
    assert got[NightIndex(0)]['o1'] == 45
    assert 'o1' not in got.get(NightIndex(1), {})


def test_empty_input():
    assert cumulative_remaining_by_night({}) == {}
