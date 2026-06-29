# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

"""An observation with no explicit timing windows must be bounded by its
program's active period (legacy ``process_timing_windows`` behaviour), not be
visible across the whole semester.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

from lucupy.minimodel import ElevationType, SkyBackground

from scheduler.services.sight._temporary.lucupy_adapters import (
    program_window,
    stage2_constraints,
)

RANGE_END = datetime(2026, 12, 31, tzinfo=timezone.utc)
PROG_START = datetime(2026, 6, 20)
PROG_END = datetime(2026, 6, 30)


def _obs(timing_windows, with_constraints=True):
    if not with_constraints:
        return SimpleNamespace(constraints=None)
    cond = SimpleNamespace(sb=SkyBackground.SBANY)
    constraints = SimpleNamespace(
        conditions=cond,
        elevation_type=ElevationType.AIRMASS,
        elevation_min=1.0,
        elevation_max=2.0,
        timing_windows=timing_windows,
    )
    return SimpleNamespace(constraints=constraints)


def test_program_window_helper_spans_program_and_is_utc():
    windows = program_window(PROG_START, PROG_END)
    assert len(windows) == 1
    assert windows[0].start == PROG_START.replace(tzinfo=timezone.utc)
    assert windows[0].end == PROG_END.replace(tzinfo=timezone.utc)


def test_program_window_helper_empty_without_dates():
    assert program_window(None, None) == []
    assert program_window(PROG_START, None) == []


def test_no_timing_windows_falls_back_to_program_window():
    c = stage2_constraints(
        _obs([]), has_resources=True, can_schedule=True, range_end=RANGE_END,
        program_start=PROG_START, program_end=PROG_END,
    )
    assert len(c.timing_windows) == 1
    assert c.timing_windows[0].start == PROG_START.replace(tzinfo=timezone.utc)
    assert c.timing_windows[0].end == PROG_END.replace(tzinfo=timezone.utc)


def test_no_constraints_falls_back_to_program_window():
    c = stage2_constraints(
        _obs(None, with_constraints=False),
        has_resources=True, can_schedule=True, range_end=RANGE_END,
        program_start=PROG_START, program_end=PROG_END,
    )
    assert len(c.timing_windows) == 1
    assert c.timing_windows[0].start == PROG_START.replace(tzinfo=timezone.utc)


def test_explicit_timing_windows_are_not_overridden():
    tw = SimpleNamespace(
        start=datetime(2026, 6, 25, tzinfo=timezone.utc),
        duration=__import__("datetime").timedelta(hours=2),
        period=None,
        repeat=0,  # NON_REPEATING
    )
    c = stage2_constraints(
        _obs([tw]), has_resources=True, can_schedule=True, range_end=RANGE_END,
        program_start=PROG_START, program_end=PROG_END,
    )
    # The single explicit 2h window is kept; the program window is NOT added.
    assert len(c.timing_windows) == 1
    assert c.timing_windows[0].start == datetime(2026, 6, 25, tzinfo=timezone.utc)
    assert c.timing_windows[0].end == datetime(2026, 6, 25, 2, 0, tzinfo=timezone.utc)


def test_no_program_dates_leaves_unrestricted():
    c = stage2_constraints(
        _obs([]), has_resources=True, can_schedule=True, range_end=RANGE_END,
    )
    assert c.timing_windows == []
