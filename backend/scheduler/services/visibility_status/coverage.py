# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""Is everything the ODB expects actually stored in the Sight DB?

A diff of two live reads — the ODB's expected set (``expected.py``) against the
observation ids present in ``visibility_data`` for one night — with the pending
set (``pending.py``) marking rows whose stored data is already known to be out
of date.

**One night at a time, by design.** A semester-wide ``GROUP BY observation_id``
over ``visibility_data`` is a few thousand observations x ~184 nights, and every
row carries JSONB ``visible_ranges`` and ``constraints``, so the heap is
multi-GB; right after a bulk-upsert run the visibility map is unset and Postgres
falls back to heap fetches. The single-night read on ``ix_visibility_night_site``
is a few thousand index entries instead.

Classification is a pure function so the part that could be *quietly* wrong is
directly testable.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Iterable, Optional

from scheduler.services import logger_factory
from scheduler.services.visibility_status.expected import (
    ExpectedObservation,
    get_expected_observations,
)
from scheduler.services.visibility_status.pending import get_pending_changes

_logger = logger_factory.create_logger(__name__)

__all__ = [
    "STATUS_STORED",
    "STATUS_PENDING",
    "STATUS_MISSING",
    "STATUS_SKIPPED",
    "ObservationCoverage",
    "CoverageSummary",
    "classify",
    "summarize",
    "paginate",
    "get_coverage_summary",
    "list_observation_coverage",
]

STATUS_STORED = "STORED"
STATUS_PENDING = "PENDING"
STATUS_MISSING = "MISSING"
STATUS_SKIPPED = "SKIPPED"


@dataclass(frozen=True)
class ObservationCoverage:
    """One observation's coverage on the night being examined."""

    observation_id: str
    program_label: str
    site: Optional[str]
    target_name: Optional[str]
    status: str
    skip_reason: Optional[str]


@dataclass(frozen=True)
class GroupCoverage:
    """Counts for one program or one site."""

    key: str
    expected: int = 0
    stored: int = 0
    pending: int = 0
    missing: int = 0
    skipped: int = 0

    @property
    def program_label(self) -> str:
        return self.key

    @property
    def site(self) -> str:
        return self.key


@dataclass(frozen=True)
class CoverageSummary:
    """Totals plus per-program and per-site breakdowns."""

    expected: int = 0
    stored: int = 0
    pending: int = 0
    missing: int = 0
    skipped: int = 0
    per_program: list[GroupCoverage] = field(default_factory=list)
    per_site: list[GroupCoverage] = field(default_factory=list)
    night_date: Optional[date] = None
    odb_read_at: Optional[datetime] = None
    pending_known: bool = True

    @property
    def is_complete(self) -> bool:
        """Nothing missing and nothing awaiting recomputation.

        Skipped observations are excluded — they can never be stored, so
        counting them would make "complete" unreachable forever.
        """
        return self.missing == 0 and self.pending == 0


def classify(
    expected: Iterable[ExpectedObservation],
    stored_ids: set[str],
    pending_ids: set[str],
) -> list[ObservationCoverage]:
    """Label each expected observation against what is stored and pending.

    Order of precedence matters. SKIPPED wins outright: those observations can
    never be stored, so reporting them as MISSING would be a permanent false
    alarm that trains operators to ignore real gaps. PENDING then wins over
    STORED, because stored-but-changed data no longer reflects the ODB.
    """
    rows: list[ObservationCoverage] = []
    for observation in expected:
        if observation.skip_reason is not None:
            status = STATUS_SKIPPED
        elif observation.observation_id in pending_ids:
            status = STATUS_PENDING
        elif observation.observation_id in stored_ids:
            status = STATUS_STORED
        else:
            status = STATUS_MISSING
        rows.append(ObservationCoverage(
            observation_id=observation.observation_id,
            program_label=observation.program_label,
            site=observation.site,
            target_name=observation.target_name,
            status=status,
            skip_reason=observation.skip_reason,
        ))
    return rows


def _tally(rows: Iterable[ObservationCoverage], key) -> list[GroupCoverage]:
    counts: dict[str, dict] = defaultdict(
        lambda: {"expected": 0, "stored": 0, "pending": 0, "missing": 0,
                 "skipped": 0}
    )
    for row in rows:
        bucket = counts[key(row) or "—"]
        bucket["expected"] += 1
        bucket[row.status.lower()] += 1
    return [
        GroupCoverage(key=name, **values)
        for name, values in sorted(counts.items())
    ]


def summarize(
    rows: list[ObservationCoverage],
    night_date: Optional[date] = None,
    odb_read_at: Optional[datetime] = None,
    pending_known: bool = True,
) -> CoverageSummary:
    """Roll classified rows up into totals and breakdowns."""
    totals = {"stored": 0, "pending": 0, "missing": 0, "skipped": 0}
    for row in rows:
        totals[row.status.lower()] += 1
    return CoverageSummary(
        expected=len(rows),
        **totals,
        per_program=_tally(rows, lambda r: r.program_label),
        per_site=_tally(rows, lambda r: r.site),
        night_date=night_date,
        odb_read_at=odb_read_at,
        pending_known=pending_known,
    )


def paginate(
    rows: list[ObservationCoverage],
    status: Optional[str] = None,
    site: Optional[str] = None,
    program_label: Optional[str] = None,
    search: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> tuple[list[ObservationCoverage], int]:
    """Filter, order and slice. Returns the page and the unpaginated total."""
    matched = [
        row for row in rows
        if (status is None or row.status == status)
        and (site is None or row.site == site)
        and (program_label is None or row.program_label == program_label)
        and (search is None or search.lower() in row.observation_id.lower())
    ]
    # Stable order, so paging does not shuffle rows between requests.
    matched.sort(key=lambda r: r.observation_id)
    total = len(matched)
    page = matched[offset:] if limit is None else matched[offset:offset + limit]
    return page, total


async def _coverage_rows(
    session, night_date: Optional[date]
) -> tuple[list[ObservationCoverage], date, datetime, bool]:
    """The shared read: expected set, stored ids for the night, pending set."""
    from scheduler.services.sight.calculator.constants import SITE_KEY_TO_ID
    from scheduler.services.sight.database.repositories import (
        NightEventRepository,
        VisibilityDataRepository,
    )
    from scheduler.services.visibility_status.tonight import current_night_date

    now = datetime.now(timezone.utc)
    expected = await get_expected_observations()
    odb_read_at = datetime.now(timezone.utc)

    night_repo = NightEventRepository(session)
    if night_date is None:
        # Sites can be on different nights at the same instant; GN is the
        # anchor for the shared view, and the per-site tabs resolve their own.
        night_date = await current_night_date(
            night_repo, SITE_KEY_TO_ID["GN"], now
        )

    visibility_repo = VisibilityDataRepository(session)
    stored_ids = await visibility_repo.get_stored_observation_ids_on_night(
        night_date
    )

    pending = await get_pending_changes(session, now=now)

    rows = classify(expected, stored_ids, set(pending.labels))
    _logger.info(
        f"Coverage for {night_date}: {len(expected)} expected, "
        f"{len(stored_ids)} stored, {len(pending.labels)} pending "
        f"(known={pending.known})."
    )
    return rows, night_date, odb_read_at, pending.known


async def get_coverage_summary(
    session, night_date: Optional[date] = None
) -> CoverageSummary:
    """Totals and breakdowns for a night. Reads the ODB live on every call."""
    rows, night_date, odb_read_at, pending_known = await _coverage_rows(
        session, night_date
    )
    return summarize(rows, night_date, odb_read_at, pending_known)


@dataclass(frozen=True)
class ObservationCoveragePage:
    observations: list[ObservationCoverage]
    total: int
    night_date: date
    odb_read_at: datetime


async def list_observation_coverage(
    session,
    night_date: Optional[date] = None,
    status: Optional[str] = None,
    site: Optional[str] = None,
    program_label: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> ObservationCoveragePage:
    """One filtered, paginated page of per-observation coverage."""
    rows, night_date, odb_read_at, _ = await _coverage_rows(session, night_date)
    page, total = paginate(
        rows,
        status=status,
        site=site,
        program_label=program_label,
        search=search,
        limit=limit,
        offset=offset,
    )
    return ObservationCoveragePage(
        observations=page,
        total=total,
        night_date=night_date,
        odb_read_at=odb_read_at,
    )
