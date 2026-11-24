# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""
Read back the plans :mod:`scheduler.services.plan_archive` stored, for the UI's plan history.

Listing a night never reads the plans themselves (about 100 KB each): only one plan's
content is fetched, when it is opened.
"""

from dataclasses import dataclass
from datetime import date
from typing import Any, List, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from scheduler.services.sight.database.models import GeneratedPlan

__all__ = ['PlanNight', 'list_plan_nights', 'list_night_plans', 'get_plan_content']

# Every column but the plan, in the order the summaries are built from.
_SUMMARY_COLUMNS = [column for column in GeneratedPlan.__table__.columns if column.name != 'plan']


@dataclass(frozen=True)
class PlanNight:
    night_date: date
    plan_count: int


async def list_plan_nights(session: AsyncSession, limit: int = 60, mode: Optional[str] = None) -> List[PlanNight]:
    """The most recent nights with stored plans, newest first."""
    stmt = (
        select(GeneratedPlan.night_date, func.count())
        .where(GeneratedPlan.night_date.is_not(None))
        .group_by(GeneratedPlan.night_date)
        .order_by(GeneratedPlan.night_date.desc())
        .limit(limit)
    )
    if mode is not None:
        stmt = stmt.where(GeneratedPlan.mode == mode)
    rows = (await session.execute(stmt)).all()
    return [PlanNight(night_date=night, plan_count=count) for night, count in rows]


async def list_night_plans(session: AsyncSession, night_date: date, mode: Optional[str] = None) -> List[Any]:
    """The plans stored for a night, oldest first, without their content."""
    stmt = select(*_SUMMARY_COLUMNS).where(GeneratedPlan.night_date == night_date).order_by(GeneratedPlan.created_at)
    if mode is not None:
        stmt = stmt.where(GeneratedPlan.mode == mode)
    return list((await session.execute(stmt)).all())


async def get_plan_content(session: AsyncSession, plan_id: int) -> Optional[dict]:
    """One stored plan, ``{nightPlans, plansSummary}`` as the UI received it. None if there is no such plan."""
    return (await session.execute(select(GeneratedPlan.plan).where(GeneratedPlan.id == plan_id))).scalar_one_or_none()
