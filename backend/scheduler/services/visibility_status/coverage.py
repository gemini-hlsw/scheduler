# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause


import asyncio
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from typing import Iterable, Optional

from scheduler.services import logger_factory
from scheduler.services.visibility_status.expected import (
    ExpectedObservation,
    get_expected_observations,
)
from scheduler.services.visibility_status.pending import get_pending_changes
from scheduler.services.visibility_status.reasons import (
    ODB_CHANGED,
    SightNightState,
    missing_reason,
)

_logger = logger_factory.create_logger(__name__)

__all__ = [
    "STATUS_STORED",
    "STATUS_PENDING",
    "STATUS_MISSING",
    "STATUS_SKIPPED",
    "ObservationCoverage",
    "CoverageSummary",
    "classify",
    "explain",
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
    """One observation's coverage on the night being examined.

    ``reason`` explains the status in one token (see ``reasons.py``): why a
    SKIPPED observation cannot be stored, why a PENDING one is out of date, or
    how far the pipeline got for a MISSING one. STORED rows have none.
    """

    observation_id: str
    program_label: str
    site: Optional[str]
    target_name: Optional[str]
    status: str
    reason: Optional[str]


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

    MISSING rows come back without a reason: ``explain`` fills those in from the
    Sight DB, keeping this function answerable from the ODB read alone.
    """
    rows: list[ObservationCoverage] = []
    for observation in expected:
        reason: Optional[str] = None
        if observation.skip_reason is not None:
            status = STATUS_SKIPPED
            reason = observation.skip_reason
        elif observation.observation_id in pending_ids:
            status = STATUS_PENDING
            reason = ODB_CHANGED
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
            reason=reason,
        ))
    return rows


def explain(
    rows: Iterable[ObservationCoverage],
    state: Optional[SightNightState],
) -> list[ObservationCoverage]:
    """Give every MISSING row the reason its pipeline stage implies.

    Pure, and rows of any other status are returned untouched.
    """
    return [
        replace(row, reason=missing_reason(row.target_name, row.site, state))
        if row.status == STATUS_MISSING else row
        for row in rows
    ]


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


async def _sight_night_state(
    session,
    night_date: date,
    rows: Iterable[ObservationCoverage],
) -> Optional[SightNightState]:
    """Ask the Sight DB how far it got for this night, for the MISSING rows.

    Scoped to the observations that need explaining, and to ids rather than
    rows, so the cost is a handful of index reads even on a night where
    everything is missing. Returns None when nothing is missing, and when a read
    fails: a reason is worth a few queries, never a failed coverage answer, so
    this degrades the way ``pending.py`` does.
    """
    from scheduler.services.sight.calculator.constants import SITE_KEY_TO_ID
    from scheduler.services.sight.database.repositories import (
        NightEventRepository,
        TargetNightDataRepository,
        TargetRepository,
    )

    missing = [row for row in rows if row.status == STATUS_MISSING]
    if not missing:
        return None

    try:
        site_ids = {
            site: SITE_KEY_TO_ID[site]
            for site in {row.site for row in missing if row.site}
            if site in SITE_KEY_TO_ID
        }
        night_repo = NightEventRepository(session)
        nights_computed = {
            site for site, site_id in site_ids.items()
            if await night_repo.exists(site_id, night_date)
        }

        names = sorted({row.target_name for row in missing if row.target_name})
        ids_by_name = await TargetRepository(session).get_ids_by_names(names)
        name_by_id = {id_: name for name, id_ in ids_by_name.items()}

        data_repo = TargetNightDataRepository(session)
        stage1_ready: set[tuple[str, str]] = set()
        # Only sites with night events: without them there is no Stage 1
        # either, so the rest are not worth a query.
        for site in nights_computed:
            ready = await data_repo.get_target_ids_on_night(
                site_ids[site], night_date, list(ids_by_name.values())
            )
            stage1_ready.update(
                (name_by_id[id_], site) for id_ in ready if id_ in name_by_id
            )
    except Exception as exc:
        _logger.warning(
            f"Could not read Sight state for {night_date}; missing rows will "
            f"go unexplained: {exc}"
        )
        return None

    return SightNightState(
        nights_computed=frozenset(nights_computed),
        targets_known=frozenset(ids_by_name),
        stage1_ready=frozenset(stage1_ready),
    )


async def _coverage_rows(
    night_date: Optional[date],
) -> tuple[list[ObservationCoverage], date, datetime, bool]:
    """The shared read: expected set, stored ids for the night, pending set.

    Owns its database session: the read is shared between concurrent callers
    (see ``_shared_coverage_rows``), so it outlives whichever request started
    it and cannot borrow that request's session.
    """
    from scheduler.services.sight.calculator.constants import SITE_KEY_TO_ID
    from scheduler.services.sight.database.connection import session_scope
    from scheduler.services.sight.database.repositories import (
        NightEventRepository,
        VisibilityDataRepository,
    )
    from scheduler.services.visibility_status.tonight import current_night_date

    now = datetime.now(timezone.utc)
    # The ODB sweep takes seconds, so it runs before a session is opened rather
    # than holding a database connection while it waits on the network.
    expected = await get_expected_observations()
    odb_read_at = datetime.now(timezone.utc)

    async with session_scope() as session:
        night_repo = NightEventRepository(session)
        if night_date is None:
            # Sites can be on different nights at the same instant; GN is the
            # anchor for the shared view, and the per-site tabs resolve their
            # own.
            night_date = await current_night_date(
                night_repo, SITE_KEY_TO_ID["GN"], now
            )

        visibility_repo = VisibilityDataRepository(session)
        stored_ids = await visibility_repo.get_stored_observation_ids_on_night(
            night_date
        )

        pending = await get_pending_changes(session, now=now)

        rows = classify(expected, stored_ids, set(pending.labels))
        rows = explain(rows, await _sight_night_state(session, night_date, rows))
    _logger.info(
        f"Coverage for {night_date}: {len(expected)} expected, "
        f"{len(stored_ids)} stored, {len(pending.labels)} pending "
        f"(known={pending.known})."
    )
    return rows, night_date, odb_read_at, pending.known


# Reads currently in flight, by night. Empty between reads: this coalesces
# concurrent callers, it does not cache results.
_reads_in_flight: dict[Optional[date], "asyncio.Task"] = {}


async def _shared_coverage_rows(
    night_date: Optional[date],
) -> tuple[list[ObservationCoverage], date, datetime, bool]:
    """``_coverage_rows``, with concurrent callers folded into one read.

    The Visibility tab mounts its summary and its list together, so both
    coverage fields arrive as separate requests milliseconds apart, and the ODB
    sweep takes seconds. A caller that arrives while a read for the same night
    is in flight awaits that read instead of starting its own and gets the same
    rows, so the two halves of the screen cannot disagree about the pending set.

    Nothing is cached once a read finishes, so an operator hitting refresh still
    reads the ODB live. The read is shielded because it is shared: a browser
    disconnecting must not cancel work another request is waiting on.
    """
    loop = asyncio.get_running_loop()
    task = _reads_in_flight.get(night_date)
    if task is None or task.get_loop() is not loop:
        task = loop.create_task(_coverage_rows(night_date))
        _reads_in_flight[night_date] = task

        def _forget(finished: "asyncio.Task", key=night_date) -> None:
            # A done-callback rather than try/finally: the entry has to go even
            # if the request that started the read was cancelled.
            if _reads_in_flight.get(key) is finished:
                del _reads_in_flight[key]

        task.add_done_callback(_forget)
    return await asyncio.shield(task)


async def get_coverage_summary(
    night_date: Optional[date] = None,
) -> CoverageSummary:
    """Totals and breakdowns for a night. Reads the ODB live on every call."""
    rows, night_date, odb_read_at, pending_known = await _shared_coverage_rows(
        night_date
    )
    return summarize(rows, night_date, odb_read_at, pending_known)


@dataclass(frozen=True)
class ObservationCoveragePage:
    observations: list[ObservationCoverage]
    total: int
    night_date: date
    odb_read_at: datetime


async def list_observation_coverage(
    night_date: Optional[date] = None,
    status: Optional[str] = None,
    site: Optional[str] = None,
    program_label: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> ObservationCoveragePage:
    """One filtered, paginated page of per-observation coverage.

    Filtering and pagination are per-request; the underlying read is shared with
    a concurrent summary request for the same night.
    """
    rows, night_date, odb_read_at, _ = await _shared_coverage_rows(night_date)
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
