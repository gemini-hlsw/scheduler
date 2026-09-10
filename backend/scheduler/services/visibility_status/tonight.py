# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""What is visible on a given night at a given site, and for how long.

Stage 2 stores visibility as index pairs into the night's one-minute grid whose
origin is ``night_events.night_start`` (see
``sight/calculations/night_events.py``), so index *i* is ``night_start + i``
minutes. Ranges are inclusive at both ends: ``[a, b]`` covers ``b - a + 1``
minutes.

"Tonight" is resolved per site from ``night_events``, not from a UTC calendar
date — GN and GS roll into a new night at different instants, so one shared date
would be wrong for one of them for part of every day.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional, Sequence

from scheduler.services import logger_factory

_logger = logger_factory.create_logger(__name__)

__all__ = [
    "VisibleInterval",
    "VisibleObservation",
    "ranges_to_intervals",
    "remaining_minutes_from_now",
    "current_night_date",
    "build_visible_observation",
]


@dataclass(frozen=True)
class VisibleInterval:
    """A contiguous window of visibility, in UTC."""

    start: datetime
    end: datetime


@dataclass(frozen=True)
class VisibleObservation:
    """One observation's visibility on one night at one site."""

    observation_id: str
    site: str
    target_name: Optional[str]
    night_date: date
    remaining_minutes: int
    remaining_minutes_from_now: int
    intervals: list[VisibleInterval]


def ranges_to_intervals(
    visible_ranges: Optional[Sequence], night_start: datetime
) -> list[VisibleInterval]:
    """Convert Stage-2 index pairs into UTC intervals.

    ``visible_ranges`` comes straight out of a JSONB column, so a malformed
    entry is skipped rather than allowed to fail the whole query — one bad row
    should not blank the tab.
    """
    intervals: list[VisibleInterval] = []
    for entry in visible_ranges or []:
        try:
            start_index, end_index = entry
            start_index = int(start_index)
            end_index = int(end_index)
        except (TypeError, ValueError):
            _logger.warning(f"Skipping malformed visible_range entry {entry!r}.")
            continue
        if end_index < start_index:
            _logger.warning(f"Skipping inverted visible_range entry {entry!r}.")
            continue
        intervals.append(VisibleInterval(
            start=night_start + timedelta(minutes=start_index),
            # Inclusive of end_index, so the window runs to the end of that
            # minute.
            end=night_start + timedelta(minutes=end_index + 1),
        ))
    return intervals


def remaining_minutes_from_now(
    intervals: Sequence[VisibleInterval], now: datetime
) -> int:
    """Minutes of visibility still ahead of ``now``.

    Windows already past contribute nothing; the one in progress contributes
    only its unelapsed part. This is the number an operator actually acts on
    mid-night, as opposed to the whole-night total.
    """
    total = timedelta()
    for interval in intervals:
        if interval.end <= now:
            continue
        total += interval.end - max(interval.start, now)
    return int(total.total_seconds() // 60)


async def current_night_date(
    night_repo, site_id: int, now: Optional[datetime] = None
) -> date:
    """The night to show for a site: the one in progress, else the next one.

    During the day no window contains ``now``, and the useful answer is the
    upcoming night — someone opening the tab at noon wants tonight, not last
    night. Falls back to the UTC date when the Sight DB has no nights for the
    site at all, so an unpopulated DB still answers instead of erroring.
    """
    now = now or datetime.now(timezone.utc)

    current = await night_repo.get_window_containing(site_id, now)
    if current is not None:
        return current.night_date

    upcoming = await night_repo.get_next_after(site_id, now)
    if upcoming is not None:
        return upcoming.night_date

    _logger.info(
        f"No night_events for site {site_id} around {now.isoformat()}; "
        f"falling back to the UTC date."
    )
    return now.date()


def build_visible_observation(
    row,
    night_start: datetime,
    site: str,
    target_name: Optional[str],
    now: datetime,
) -> VisibleObservation:
    """Turn one visibility_data row into a UI-ready record."""
    intervals = ranges_to_intervals(row.visible_ranges, night_start)
    return VisibleObservation(
        observation_id=row.observation_id,
        site=site,
        target_name=target_name,
        night_date=row.night_date,
        remaining_minutes=int(row.remaining_minutes),
        remaining_minutes_from_now=remaining_minutes_from_now(intervals, now),
        intervals=intervals,
    )


@dataclass(frozen=True)
class VisibleObservationsPage:
    """One page of a site's visible observations, plus the site's totals."""

    site: str
    night_date: date
    observations: list[VisibleObservation]
    total: int
    total_remaining_minutes: int


async def list_visible_observations(
    session,
    site: str,
    night_date: Optional[date] = None,
    limit: int = 50,
    offset: int = 0,
    min_remaining_minutes: int = 1,
    now: Optional[datetime] = None,
) -> VisibleObservationsPage:
    """Visible observations for one site on one night, paginated.

    ``night_date`` defaults to the site's current night. ``min_remaining_minutes``
    defaults to 1 so observations that are never visible are left out — the
    question is what *can* be observed.
    """
    from scheduler.services.sight.calculator.constants import SITE_KEY_TO_ID
    from scheduler.services.sight.database.repositories import (
        NightEventRepository,
        VisibilityDataRepository,
    )

    now = now or datetime.now(timezone.utc)
    site_id = SITE_KEY_TO_ID[site]
    night_repo = NightEventRepository(session)
    visibility_repo = VisibilityDataRepository(session)

    if night_date is None:
        night_date = await current_night_date(night_repo, site_id, now)

    night_event = await night_repo.get_by_site_and_night(site_id, night_date)
    if night_event is None:
        # Nothing computed for that night yet: an empty page is the honest
        # answer, and it keeps the tab usable against a partially filled DB.
        _logger.info(
            f"No night_events row for site {site} on {night_date}; "
            f"returning an empty page."
        )
        return VisibleObservationsPage(
            site=site,
            night_date=night_date,
            observations=[],
            total=0,
            total_remaining_minutes=0,
        )

    rows = await visibility_repo.get_visible_on_night(
        night_date=night_date,
        site_id=site_id,
        min_remaining_minutes=min_remaining_minutes,
        limit=limit,
        offset=offset,
    )
    total = await visibility_repo.count_visible_on_night(
        night_date=night_date,
        site_id=site_id,
        min_remaining_minutes=min_remaining_minutes,
    )

    observations = [
        build_visible_observation(
            row, night_event.night_start, site, target_name, now
        )
        for row, target_name in rows
    ]
    return VisibleObservationsPage(
        site=site,
        night_date=night_date,
        observations=observations,
        total=total,
        # Page-scoped, and labelled as such in the UI: summing the whole night
        # would mean pulling every row back just to add them up.
        total_remaining_minutes=sum(
            o.remaining_minutes_from_now for o in observations
        ),
    )
