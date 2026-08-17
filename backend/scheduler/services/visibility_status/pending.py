# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""Observations whose stored visibility is already out of date.

The ODB reports entities whose visibility inputs changed since a given time.
Anything changed since the aggregator's last successful run has stored data that
no longer reflects the ODB, so the UI shows it as *being updated* rather than as
correctly stored.

**Strictly read-only.** It reads the aggregator's change watermark and never
writes it: the watermark is how the aggregator knows which window it still has
to apply, so advancing it from here would make the aggregator skip changes it
never processed.

Unlike the expected-set sweep this degrades instead of raising — a failure costs
a badge, not a wrong "nothing is missing" verdict — matching the fail-open style
of ``coordination.get_aggregator_status``.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from scheduler.config import config
from scheduler.clients.gpp import gpp
from scheduler.services import logger_factory
from scheduler.services.visibility_aggregator import coordination
from scheduler.services.visibility_aggregator.aggregator import (
    resolve_schedulable_observation_labels,
)

_logger = logger_factory.create_logger(__name__)

__all__ = ["PendingChanges", "get_pending_changes"]


@dataclass(frozen=True)
class PendingChanges:
    """Reference labels whose visibility inputs changed since ``since``.

    ``known`` is False when the ODB could not be reached or a resolution chunk
    failed, so the UI can say "unknown" rather than presenting an empty set as
    "nothing pending".
    """

    labels: frozenset[str]
    since: Optional[datetime]
    known: bool

    @classmethod
    def unknown(cls, since: Optional[datetime] = None) -> "PendingChanges":
        return cls(labels=frozenset(), since=since, known=False)


async def _watermark(session) -> Optional[datetime]:
    """The aggregator's change watermark, or None if unavailable.

    A missing coordination row (or an un-migrated table) is not an error here —
    the caller falls back to a lookback window.
    """
    try:
        return await coordination.get_change_watermark(session)
    except Exception as exc:
        _logger.debug(f"Could not read the change watermark: {exc}")
        return None


async def get_pending_changes(
    session, now: Optional[datetime] = None
) -> PendingChanges:
    """Observations the ODB reports as changed since the last successful run."""
    now = now or datetime.now(timezone.utc)
    lookback = timedelta(
        hours=float(config.visibility_aggregator.changes_fallback_lookback_hours)
    )
    since = await _watermark(session) or (now - lookback)

    try:
        changes = await gpp.client.scheduler.get_visibility_changes(since)
    except Exception as exc:
        _logger.warning(
            f"Could not read visibility changes since {since.isoformat()}: {exc}"
        )
        return PendingChanges.unknown(since)

    internal_ids = sorted(changes.observation_ids)
    if not internal_ids:
        return PendingChanges(labels=frozenset(), since=since, known=True)

    try:
        resolution = await resolve_schedulable_observation_labels(internal_ids)
    except Exception as exc:
        _logger.warning(f"Could not resolve changed observations: {exc}")
        return PendingChanges.unknown(since)

    if resolution.failed_chunks:
        # Some ids never came back, so the set is incomplete. Report it as
        # unknown rather than as a short but authoritative list.
        _logger.warning(
            f"{resolution.failed_chunks} chunk(s) failed while resolving "
            f"{len(internal_ids)} changed observations; reporting pending as "
            f"unknown."
        )
        return PendingChanges.unknown(since)

    # Ids that do not come back are deleted, non-schedulable, or unlabelled —
    # nothing stored to invalidate, so they are simply dropped.
    return PendingChanges(
        labels=frozenset(resolution.labels.values()), since=since, known=True
    )
