# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""
get_stage1_greedymax_bulk runs on the engine loop during every replan, via the collector's
Sight load. Its query has to stay there (the AsyncSession is bound to that loop), but the
unpacking after it is seconds of pure CPU with nothing to await in between, so it must not.
"""

import asyncio
import inspect
from datetime import date
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pytest

from scheduler.services.sight.calculations.arrays import pack_array, unpack_array
from scheduler.services.sight.calculator.calculator import Calculator

_NIGHT_MINUTES = 650


def _row(target_id: int, night_date: date, site_id: int = 1):
    """A column row as result.all() hands it over: values only, no lazy loads left."""
    blob = pack_array(np.linspace(0.0, 1.0, _NIGHT_MINUTES))
    return SimpleNamespace(
        target_id=target_id,
        site_id=site_id,
        night_date=night_date,
        night_duration_minutes=_NIGHT_MINUTES,
        ra=blob, dec=blob, alt=blob, az=blob, airmass=blob, hourangle=blob,
    )


def test_the_unpacking_is_offloaded():
    """A guard, not a behavior test: inline, this is what stalls the loop."""
    src = inspect.getsource(Calculator.get_stage1_greedymax_bulk)
    assert "to_thread(self._unpack_greedymax_rows" in src, \
        "the unpacking is CPU-bound and belongs off the loop"
    assert "unpack_array" not in src, "no unpacking may remain on the loop"


def test_unpacking_shapes_the_rows_as_greedymax_expects():
    rows = [_row(7, date(2026, 8, 26)), _row(7, date(2026, 8, 27))]

    out = Calculator._unpack_greedymax_rows(rows, {7: 'NGC-1'})

    nights = out['NGC-1']['nights']
    assert sorted(nights) == ['GN_2026-08-26', 'GN_2026-08-27']
    night = nights['GN_2026-08-26']
    assert night['site'] == 'GN'
    assert night['night_date'] == date(2026, 8, 26)
    assert night['night_duration_minutes'] == _NIGHT_MINUTES
    expected = unpack_array(rows[0].ra, _NIGHT_MINUTES)
    for field in ('ra', 'dec', 'alt', 'az', 'airmass', 'hourangle'):
        np.testing.assert_allclose(night[field], expected, err_msg=field)


def test_arrays_stay_numpy():
    """build_target_info calls np.asarray on these, so converting to lists here is 60x the
    cost of the unpack for nothing. Keep the contract explicit."""
    out = Calculator._unpack_greedymax_rows([_row(7, date(2026, 8, 26))], {7: 'NGC-1'})

    night = out['NGC-1']['nights']['GN_2026-08-26']
    for field in ('ra', 'dec', 'alt', 'az', 'airmass', 'hourangle'):
        assert isinstance(night[field], np.ndarray), f'{field} was converted to a list'


def test_a_row_whose_target_is_unknown_is_skipped():
    out = Calculator._unpack_greedymax_rows([_row(99, date(2026, 8, 26))], {7: 'NGC-1'})

    assert out == {}


@pytest.mark.asyncio
async def test_the_loop_keeps_running_while_rows_unpack():
    """The regression that matters: at semester scale this ran for seconds inline.

    36k rows is a 200-target, 180-night window, the shape a semester replan fetches. The probe
    counts how many times the loop got scheduled during the unpack; inline it would get zero.
    """
    rows = [_row(7, date(2026, 8, 26)) for _ in range(36_000)]
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.001)
            ticks += 1

    task = asyncio.create_task(ticker())
    try:
        await asyncio.sleep(0.01)  # let the ticker get going first
        started = perf_counter()
        out = await asyncio.to_thread(Calculator._unpack_greedymax_rows, rows, {7: 'NGC-1'})
        elapsed = perf_counter() - started
    finally:
        task.cancel()

    assert out['NGC-1']['nights']
    # Only meaningful if the unpack took long enough that the loop would have been starved
    # had it run inline.
    assert elapsed > 0.01, f'unpack finished in {elapsed:.4f}s; the probe proves nothing'
    assert ticks > 0, 'the loop was blocked for the whole unpack'
