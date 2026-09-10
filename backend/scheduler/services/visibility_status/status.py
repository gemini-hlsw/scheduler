# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""Typed view of the aggregator's coordination row.

``coordination.get_aggregator_status`` returns the run's progress as an opaque
JSON string. This turns it into typed fields — phase, progress pair, ETA — so
the UI is handed values rather than a blob to decode.

The payload is written by whichever aggregator version happens to be deployed,
which can be older or newer than this reader, so every field is parsed
defensively: missing keys, wrong types and unparseable JSON all degrade to None.
A status card is not worth failing a query over.
"""

import json
from dataclasses import dataclass
from typing import Any, Optional

from scheduler.services import logger_factory
from scheduler.services.visibility_aggregator import coordination

_logger = logger_factory.create_logger(__name__)

__all__ = ["AggregatorStatus", "get_aggregator_status"]


@dataclass(frozen=True)
class AggregatorStatus:
    """Current state of the background visibility-aggregator run."""

    active: bool
    stale: bool
    holder: Optional[str]
    started_at: Optional[str]
    heartbeat_at: Optional[str]
    finished_at: Optional[str]
    phase: Optional[str]
    progress_current: Optional[int]
    progress_total: Optional[int]
    progress_unit: Optional[str]
    elapsed_seconds: Optional[float]
    eta_seconds: Optional[float]
    detail: Optional[str]


def _parse_detail(raw: Optional[str]) -> dict:
    """The heartbeat payload as a dict, or empty if it cannot be read."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError) as exc:
        _logger.debug(f"Unparseable aggregator detail {raw!r}: {exc}")
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _as(kind, value: Any):
    """Coerce to ``kind``, or None if the payload holds something else.

    bool is rejected explicitly: it is an int subclass in Python, so a stray
    ``true`` would otherwise surface as a progress count of 1.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return kind(value)
    except (TypeError, ValueError):
        return None


async def get_aggregator_status() -> AggregatorStatus:
    """Snapshot the coordination row with its progress detail parsed out."""
    row = await coordination.get_aggregator_status()
    detail = _parse_detail(row.get("detail"))
    phase = detail.get("phase")

    return AggregatorStatus(
        active=bool(row.get("active")),
        stale=bool(row.get("stale")),
        holder=row.get("holder"),
        # The run's own start, when it reported one, else the coordination row's.
        started_at=detail.get("started_at") or row.get("started_at"),
        heartbeat_at=row.get("heartbeat_at"),
        finished_at=row.get("finished_at"),
        phase=str(phase) if isinstance(phase, str) else None,
        progress_current=_as(int, detail.get("progress_current")),
        progress_total=_as(int, detail.get("progress_total")),
        progress_unit=(
            detail.get("progress_unit")
            if isinstance(detail.get("progress_unit"), str) else None
        ),
        elapsed_seconds=_as(float, detail.get("elapsed_seconds")),
        eta_seconds=_as(float, detail.get("eta_seconds")),
        detail=row.get("detail"),
    )
