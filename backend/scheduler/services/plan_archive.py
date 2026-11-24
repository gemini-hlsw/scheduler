# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""
Store every plan the real-time engine generates in the Sight DB (``generated_plans``).

Each row has the event that triggered the plan and the plan itself as JSONB, in the shape
the UI receives it (camelCase, ``{nightPlans, plansSummary}``), so a stored plan reads like
the ``queueSchedule`` subscription and needs no migration when the plan format changes.

Storing is best-effort and never delays a plan: writes run as background tasks, and a
missing DATABASE_URL, an unmigrated table or a DB error is only logged. Writes run one at
a time in the order they were requested, so a decision always finds its proposal's row.
"""

import asyncio
import dataclasses
import math
import os
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional, Set

from sqlalchemy import insert, update
from strawberry.types.base import StrawberryContainer

from scheduler.config import config
from scheduler.core.builder import modes
from scheduler.services import logger_factory
from scheduler.services.sight.database.connection import session_scope
from scheduler.services.sight.database.models import GeneratedPlan
from scheduler.version import get_app_version

__all__ = ['PlanArchive', 'PlanStatus', 'to_json']

_logger = logger_factory.create_logger(__name__)


class PlanStatus(str, Enum):
    IN_EFFECT = 'in_effect'
    PROPOSED = 'proposed'
    ACCEPTED = 'accepted'
    DISCARDED = 'discarded'
    # Replaced by a newer proposal before anyone decided on it.
    SUPERSEDED = 'superseded'
    # Dropped without a decision: the night ended, or the next plan failed.
    EXPIRED = 'expired'


def _enabled() -> bool:
    section = getattr(config, 'plan_archive', None)
    return bool(getattr(section, 'enabled', True)) if section is not None else True


def _app_version() -> str:
    # The deployed APP_VERSION also names the server (e.g. 1.4.0-staging-simulation), which the
    # version file does not: the realtime and simulation servers share the database.
    return os.environ.get('APP_VERSION') or get_app_version()


def _night_date(night_plans: dict) -> Optional[date]:
    """
    The night a plan covers, as the UI labels it: the UTC date of its morning twilight.
    None when the plan has no night in it.
    """
    try:
        morning = night_plans['nightTimeline'][0]['timeEntriesBySite'][0]['mornTwilight']
        return datetime.fromisoformat(morning).astimezone(timezone.utc).date()
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _to_camel_case(name: str) -> str:
    head, *rest = name.split('_')
    return head + ''.join(part[:1].upper() + part[1:] for part in rest)


def _custom_scalar_serializer(field_type: Any) -> Optional[Callable[[Any], Any]]:
    """The serialize function of a custom scalar field (e.g. SObservationID), through lists and Optional."""
    while isinstance(field_type, StrawberryContainer):
        field_type = field_type.of_type
    definition = getattr(field_type, '_scalar_definition', None)
    return definition.serialize if definition is not None else None


def to_json(value: Any, serialize: Optional[Callable[[Any], Any]] = None) -> Any:
    """
    The GraphQL output for a Strawberry value, as plain JSON: field names in camelCase, custom
    scalars through their serialize function, enums by name, datetimes in ISO format. JSON
    scalars (summaries) are kept as they are, with their keys as strings. Non-finite floats
    become null, which JSONB has no other way to hold.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_json(item, serialize) for item in value]
    if serialize is not None:
        return to_json(serialize(value))
    definition = getattr(type(value), '__strawberry_definition__', None)
    if definition is not None:
        return {
            (field.graphql_name or _to_camel_case(field.python_name)): to_json(
                getattr(value, field.python_name), _custom_scalar_serializer(field.type)
            )
            for field in definition.fields
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {_to_camel_case(field.name): to_json(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {(key.name if isinstance(key, Enum) else str(key)): to_json(item) for key, item in value.items()}
    if hasattr(value, 'item') and callable(value.item):
        # A numpy scalar.
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class PlanArchive:
    """Stores the plans of one engine. Every method returns at once; the write runs later."""

    def __init__(self, enabled: Optional[bool] = None):
        self.enabled = _enabled() if enabled is None else enabled
        # One write at a time, in order: asyncio.Lock wakes waiters first come, first served.
        self._lock = asyncio.Lock()
        # Referenced until done, or the event loop may collect a running task.
        self._tasks: Set[asyncio.Task] = set()
        self._warned = False

    def record_plan(self, event, response, status: PlanStatus, proposal_id: Optional[str] = None) -> None:
        """
        Store a plan the engine generated and the event that triggered it.

        Args:
            event: the event the plan was computed for.
            response: the NewNightPlans or ProposedNightPlans sent to the UIs.
            status: IN_EFFECT when the plan took effect at once, PROPOSED when it waits for a decision.
            proposal_id: the proposal's id, for a proposed plan.
        """
        if not self.enabled:
            return
        site = getattr(event, 'site', None)
        night_plans = to_json(response.night_plans)
        row = {
            'app_version': _app_version()[:100],
            # Read at each plan rather than imported, so it is the mode the server runs in.
            'mode': modes.app_mode.value,
            'event_type': type(event).__name__,
            'event_description': str(event.description),
            'event_time': event.time,
            'event_site': site.name if site is not None else None,
            'night_date': _night_date(night_plans),
            'status': status.value,
            'proposal_id': proposal_id,
            'plan': {'nightPlans': night_plans, 'plansSummary': to_json(response.plans_summary)},
        }
        self._spawn(self._insert(row), f'plan for "{row["event_description"]}"')

    def record_decision(self, proposal_id: str, accepted: bool, decided_by: str, decided_at: datetime) -> None:
        """Mark a proposed plan as accepted or discarded, and by whom."""
        if not self.enabled:
            return
        values = {
            'status': (PlanStatus.ACCEPTED if accepted else PlanStatus.DISCARDED).value,
            'decided_by': decided_by[:255],
            'decided_at': decided_at,
        }
        self._spawn(self._update(proposal_id, values), f'decision on plan {proposal_id}')

    def record_superseded(self, proposal_id: str) -> None:
        """Mark a proposed plan as replaced by a newer proposal before anyone decided on it."""
        if not self.enabled:
            return
        values = {'status': PlanStatus.SUPERSEDED.value}
        self._spawn(self._update(proposal_id, values), f'replacement of plan {proposal_id}')

    def record_expired(self, proposal_id: str) -> None:
        """Mark a proposed plan as dropped without a decision: the night ended, or the next plan failed."""
        if not self.enabled:
            return
        values = {'status': PlanStatus.EXPIRED.value}
        self._spawn(self._update(proposal_id, values), f'expiry of plan {proposal_id}')

    async def flush(self) -> None:
        """Wait for every write requested so far. For tests and shutdown."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    def _spawn(self, write, what: str) -> None:
        task = asyncio.get_running_loop().create_task(self._run(write, what))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run(self, write, what: str) -> None:
        async with self._lock:
            try:
                await write
            except Exception as e:
                # Once at warning level, so a server without the table or DB says so without
                # repeating it for every plan of the night.
                log = _logger.debug if self._warned else _logger.warning
                self._warned = True
                log(f'Could not store the {what} in generated_plans: {e}')

    @staticmethod
    async def _insert(row: dict) -> None:
        async with session_scope() as session:
            await session.execute(insert(GeneratedPlan).values(**row))

    @staticmethod
    async def _update(proposal_id: str, values: dict) -> None:
        async with session_scope() as session:
            await session.execute(
                update(GeneratedPlan)
                # Only a plan still waiting: a decided or superseded one keeps its outcome.
                .where(GeneratedPlan.proposal_id == proposal_id, GeneratedPlan.status == PlanStatus.PROPOSED.value)
                .values(**values)
            )
