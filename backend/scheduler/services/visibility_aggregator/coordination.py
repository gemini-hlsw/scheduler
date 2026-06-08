# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""Cross-process coordination between the operation dyno and the aggregator cron.

State lives in the ``scheduler_coordination`` Postgres table (one row per
activity) so it is visible across separate dynos. We deliberately avoid Postgres
advisory locks: a session-level lock would need a pinned connection held across
commits, which fights SQLAlchemy's connection-per-transaction pooling. Instead,
acquisition is an *atomic conditional upsert* (``INSERT ... ON CONFLICT DO UPDATE
... WHERE not active or stale``) and liveness is a committed ``heartbeat_at`` plus
a staleness threshold, so a crashed holder never wedges the interlock.

Two rows:
  - ``visibility_aggregator`` — active while the aggregator runs; the operation
    process waits on it before creating a plan.
  - ``night_execution`` — active while the operation process computes a plan; the
    aggregator skips a cycle if it is set (and fresh).
"""

import asyncio
import json
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from scheduler.config import config
from scheduler.services.logger_factory import create_logger
from scheduler.services.sight.database.connection import session_scope

_logger = create_logger(__name__, with_id=False)

__all__ = [
    "AGGREGATOR_NAME",
    "NIGHT_EXECUTION_NAME",
    "STALE_AFTER_SECONDS",
    "acquire_aggregator",
    "heartbeat_aggregator",
    "release_aggregator",
    "is_aggregator_active",
    "wait_until_aggregator_idle",
    "signal_plan_in_progress",
    "signal_plan_done",
    "is_plan_in_progress",
    "get_aggregator_status",
]

AGGREGATOR_NAME = "visibility_aggregator"
NIGHT_EXECUTION_NAME = "night_execution"

# Tunables live in config.yaml under the `visibility_aggregator` section.
_agg_config = config.visibility_aggregator

# A holder whose heartbeat is older than this is treated as dead.
STALE_AFTER_SECONDS: float = float(_agg_config.stale_after_seconds)

# Cadence the operation process polls at while blocked on a run.
_IDLE_POLL_INTERVAL_SECONDS: float = float(_agg_config.idle_poll_interval_seconds)


# --- low-level helpers (caller owns the transaction / commit) ----------------

async def _acquire(
    session: AsyncSession,
    name: str, holder: str,
    stale_after: float
) -> bool:
    """Atomically take ``name`` if it is inactive or its holder is stale.

    Returns True iff this caller now owns the row. Caller must commit.
    """
    stmt = text(
        """
        INSERT INTO scheduler_coordination
            (name, active, holder, started_at, heartbeat_at, finished_at)
        VALUES (:name, true, :holder, now(), now(), null)
        ON CONFLICT (name) DO UPDATE
            SET active = true,
                holder = :holder,
                started_at = now(),
                heartbeat_at = now(),
                finished_at = null
            WHERE scheduler_coordination.active = false
               OR scheduler_coordination.heartbeat_at
                  < now() - make_interval(secs => :stale)
        RETURNING name
        """
    )
    result = await session.execute(
        stmt, {"name": name, "holder": holder, "stale": stale_after}
    )
    return result.first() is not None


async def _heartbeat(session: AsyncSession, name: str,
                     detail: Optional[dict]) -> None:
    """Refresh ``heartbeat_at`` (and optionally ``detail``). Caller commits."""
    if detail is None:
        stmt = text(
            "UPDATE scheduler_coordination SET heartbeat_at = now() "
            "WHERE name = :name"
        )
        await session.execute(stmt, {"name": name})
    else:
        stmt = text(
            "UPDATE scheduler_coordination "
            "SET heartbeat_at = now(), detail = cast(:detail as jsonb) "
            "WHERE name = :name"
        )
        await session.execute(stmt, {"name": name, "detail": json.dumps(detail)})


async def _release(session: AsyncSession, name: str) -> None:
    """Mark ``name`` inactive. Caller commits."""
    stmt = text(
        "UPDATE scheduler_coordination "
        "SET active = false, finished_at = now(), heartbeat_at = now() "
        "WHERE name = :name"
    )
    await session.execute(stmt, {"name": name})


async def _is_active(session: AsyncSession, name: str,
                     stale_after: float) -> bool:
    """True if ``name`` is active and its heartbeat is fresh."""
    stmt = text(
        """
        SELECT 1 FROM scheduler_coordination
        WHERE name = :name
          AND active = true
          AND heartbeat_at > now() - make_interval(secs => :stale)
        """
    )
    result = await session.execute(stmt, {"name": name, "stale": stale_after})
    return result.first() is not None



async def acquire_aggregator(session: AsyncSession, holder: str,
                             stale_after: float = STALE_AFTER_SECONDS) -> bool:
    return await _acquire(session, AGGREGATOR_NAME, holder, stale_after)


async def heartbeat_aggregator(session: AsyncSession,
                               detail: Optional[dict] = None) -> None:
    await _heartbeat(session, AGGREGATOR_NAME, detail)


async def release_aggregator(session: AsyncSession) -> None:
    await _release(session, AGGREGATOR_NAME)


async def is_plan_in_progress(session: AsyncSession,
                              stale_after: float = STALE_AFTER_SECONDS) -> bool:
    return await _is_active(session, NIGHT_EXECUTION_NAME, stale_after)


async def is_aggregator_active(session: AsyncSession,
                               stale_after: float = STALE_AFTER_SECONDS) -> bool:
    return await _is_active(session, AGGREGATOR_NAME, stale_after)


def _plan_wait_timeout() -> Optional[float]:
    """Cap on how long plan creation blocks for the aggregator.

    Configured by ``visibility_aggregator.plan_wait_timeout`` in config.yaml;
    None means "block until finished" (the agreed hard interlock). That key is
    env-overridable per-deploy via ``VIS_AGG_PLAN_WAIT_TIMEOUT``.
    """
    value = config.visibility_aggregator.plan_wait_timeout
    return float(value) if value is not None else None


async def wait_until_aggregator_idle(poll_interval: float = _IDLE_POLL_INTERVAL_SECONDS,
                                     timeout: Optional[float] = None) -> bool:
    """Block until the aggregator row is inactive/stale (the hard interlock).

    Each poll uses its own short session so it sees the aggregator's committed
    heartbeats. Returns True when idle, False if ``timeout`` elapses first. The
    default (block until finished) honours the agreed interlock; pass a timeout
    or set ``VIS_AGG_PLAN_WAIT_TIMEOUT`` to cap the wait. A missing DATABASE_URL
    is treated as "no aggregator" so local/in-memory runs are unaffected.
    """
    if timeout is None:
        timeout = _plan_wait_timeout()
    waited = 0.0
    while True:
        try:
            async with session_scope() as session:
                if not await is_aggregator_active(session):
                    if waited:
                        _logger.info(
                            f"Aggregator finished after waiting {waited:.0f}s; "
                            f"proceeding with plan creation."
                        )
                    return True
        except Exception as exc:
            # Fail open: a missing DATABASE_URL, a coordination table that hasn't
            # been migrated yet (deploy window), or a transient DB error must
            # never block plan creation. (CancelledError is a BaseException and
            # is intentionally not caught here.)
            _logger.warning(
                f"Coordination check unavailable ({exc}); proceeding with plan "
                f"creation without the aggregator interlock."
            )
            return True
        if timeout is not None and waited >= timeout:
            _logger.warning(
                f"Timed out after {waited:.0f}s waiting for the visibility "
                f"aggregator to finish; proceeding with plan creation."
            )
            return False
        if waited == 0.0:
            _logger.info(
                "Visibility aggregator is running; deferring plan creation "
                "until it finishes."
            )
        await asyncio.sleep(poll_interval)
        waited += poll_interval


async def signal_plan_in_progress(holder: str = "operation",
                                  detail: Optional[dict] = None) -> None:
    """Mark that the operation process is computing a plan (best-effort)."""
    try:
        async with session_scope() as session:
            await _acquire_unconditional(session, NIGHT_EXECUTION_NAME, holder)
            if detail is not None:
                await _heartbeat(session, NIGHT_EXECUTION_NAME, detail)
    except Exception as exc:
        # Best-effort signal; never let it interfere with planning (no DB,
        # un-migrated table, or transient error).
        _logger.debug(f"Could not signal plan-in-progress: {exc}")


async def signal_plan_done() -> None:
    """Clear the plan-in-progress flag (best-effort)."""
    try:
        async with session_scope() as session:
            await _release(session, NIGHT_EXECUTION_NAME)
    except Exception as exc:
        _logger.debug(f"Could not clear plan-in-progress: {exc}")


async def get_aggregator_status() -> dict:
    """Snapshot the aggregator coordination row for display (UI / debugging).

    Timestamps are returned as ISO strings; ``stale`` is True when the row is
    marked active but its heartbeat has expired. Safe when DATABASE_URL is unset.
    """
    empty = {
        "active": False, "holder": None, "started_at": None,
        "heartbeat_at": None, "finished_at": None, "stale": False, "detail": None,
    }
    try:
        async with session_scope() as session:
            stmt = text(
                """
                SELECT active, holder, started_at, heartbeat_at, finished_at, detail,
                       (active AND heartbeat_at
                            > now() - make_interval(secs => :stale)) AS fresh
                FROM scheduler_coordination
                WHERE name = :name
                """
            )
            row = (await session.execute(
                stmt, {"name": AGGREGATOR_NAME, "stale": STALE_AFTER_SECONDS}
            )).mappings().first()
    except Exception as exc:
        # No DB / un-migrated table / transient error: report "not running"
        # rather than failing the GraphQL query.
        _logger.debug(f"Could not read aggregator status: {exc}")
        return empty
    if row is None:
        return empty

    def _iso(value):
        return value.isoformat() if value is not None else None

    return {
        "active": bool(row["active"]),
        "holder": row["holder"],
        "started_at": _iso(row["started_at"]),
        "heartbeat_at": _iso(row["heartbeat_at"]),
        "finished_at": _iso(row["finished_at"]),
        "stale": bool(row["active"]) and not bool(row["fresh"]),
        "detail": json.dumps(row["detail"]) if row["detail"] is not None else None,
    }


async def _acquire_unconditional(session: AsyncSession, name: str,
                                 holder: str) -> None:
    """Set ``name`` active unconditionally (single writer, no contention)."""
    stmt = text(
        """
        INSERT INTO scheduler_coordination
            (name, active, holder, started_at, heartbeat_at, finished_at)
        VALUES (:name, true, :holder, now(), now(), null)
        ON CONFLICT (name) DO UPDATE
            SET active = true, holder = :holder, heartbeat_at = now(),
                finished_at = null
        """
    )
    await session.execute(stmt, {"name": name, "holder": holder})
