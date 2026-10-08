# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""
The parallel aggregation cuts work into (night, target chunk) tasks. Every
(target, night) of a target's window must land in exactly one chunk: two
chunks sharing one would write the same rows from two workers, and a missing
one would leave a gap the aggregator later has to fill.
"""
from collections import Counter
from datetime import date, timedelta

from scheduler.services.visibility_aggregator.aggregate_parallel import plan_chunks

_WINDOWS = {
    "A": (date(2026, 10, 1), date(2026, 10, 3)),
    "B": (date(2026, 10, 2), date(2026, 10, 2)),
    "C": (date(2026, 10, 2), date(2026, 10, 5)),
    "D": (date(2026, 9, 1), date(2026, 12, 1)),
}


def _pairs(chunks):
    return Counter((night, name) for night, names in chunks for name in names)


def test_each_target_night_lands_in_exactly_one_chunk():
    first, last = date(2026, 10, 1), date(2026, 10, 5)
    chunks = plan_chunks(_WINDOWS, first, last, chunk_size=2)

    expected = Counter()
    for name, (start, end) in _WINDOWS.items():
        night = max(start, first)
        while night <= min(end, last):
            expected[(night, name)] += 1
            night += timedelta(days=1)
    assert _pairs(chunks) == expected


def test_chunks_respect_the_size_and_stay_within_one_night():
    chunks = plan_chunks(_WINDOWS, date(2026, 10, 2), date(2026, 10, 2), chunk_size=3)

    # A, B, C and D are all due on the 2nd: one full chunk and the remainder.
    assert chunks == [
        (date(2026, 10, 2), ("A", "B", "C")),
        (date(2026, 10, 2), ("D",)),
    ]


def test_nights_outside_the_requested_range_are_left_out():
    chunks = plan_chunks(_WINDOWS, date(2026, 10, 4), date(2026, 10, 4), chunk_size=10)

    assert chunks == [(date(2026, 10, 4), ("C", "D"))]
