from datetime import date
from typing import Sequence

from sqlalchemy import select, delete, and_, func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from scheduler.services.sight.database.models import Target, VisibilityData


# Rows per executemany chunk in bulk_upsert: bounds per-statement bind
# buffers (~1000 rows x 2-5KB of JSONB each) while keeping round trips low.
BULK_UPSERT_CHUNK = 1000


class VisibilityDataRepository:
    """Repository for VisibilityData model."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_observation(
        self,
        observation_id: str,
        start_date: date,
        end_date: date,
    ) -> Sequence[VisibilityData]:
        """Get visibility data for an observation across a date range."""
        stmt = select(VisibilityData).where(
            and_(
                VisibilityData.observation_id == observation_id,
                VisibilityData.night_date >= start_date,
                VisibilityData.night_date <= end_date,
            )
        ).order_by(VisibilityData.night_date)
        
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_by_target(
        self,
        target_id: int,
        start_date: date,
        end_date: date,
    ) -> Sequence[VisibilityData]:
        """Get visibility data for a target across a date range."""
        stmt = select(VisibilityData).where(
            and_(
                VisibilityData.target_id == target_id,
                VisibilityData.night_date >= start_date,
                VisibilityData.night_date <= end_date,
            )
        ).order_by(VisibilityData.night_date)
        
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_by_observation_and_target(
        self,
        observation_id: str,
        target_id: int,
        start_date: date,
        end_date: date,
    ) -> Sequence[VisibilityData]:
        """Get visibility data for a specific observation and target."""
        stmt = select(VisibilityData).where(
            and_(
                VisibilityData.observation_id == observation_id,
                VisibilityData.target_id == target_id,
                VisibilityData.night_date >= start_date,
                VisibilityData.night_date <= end_date,
            )
        ).order_by(VisibilityData.night_date)
        
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_by_observation_ids_on_night(
        self,
        observation_ids: list[str],
        night_date: date,
    ) -> Sequence[VisibilityData]:
        """Get visibility data for multiple observations on a single night."""
        if not observation_ids:
            return []
        stmt = select(VisibilityData).where(
            and_(
                VisibilityData.observation_id.in_(observation_ids),
                VisibilityData.night_date == night_date,
            )
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_stored_observation_nights(
        self,
        observation_ids: list[str],
        start_date: date,
        end_date: date,
    ) -> set[tuple[str, date]]:
        """
        Get the (observation_id, night_date) pairs already stored in a range.

        One dates-only query for the whole range, so a caller checking
        coverage night by night does not need a query per night, and no
        JSONB payloads are transferred. Duplicate rows for the same
        observation and night (the unique constraint also spans target and
        site) collapse into a single pair.
        """
        if not observation_ids:
            return set()
        stmt = select(
            VisibilityData.observation_id,
            VisibilityData.night_date,
        ).where(
            and_(
                VisibilityData.observation_id.in_(observation_ids),
                VisibilityData.night_date >= start_date,
                VisibilityData.night_date <= end_date,
            )
        ).distinct()
        result = await self.session.execute(stmt)
        return {(observation_id, night_date) for observation_id, night_date in result.all()}

    async def get_visible_on_night(
        self,
        night_date: date,
        site_id: int,
        min_remaining_minutes: int = 0,
        limit: int | None = None,
        offset: int = 0,
    ) -> Sequence[tuple[VisibilityData, str]]:
        """Rows for one night at one site, paired with their target name.

        Filtering on (night_date, site_id) hits ix_visibility_night_site;
        without it this scans a table holding one JSONB-carrying row per
        observation per night of the semester. The target name is joined in
        the same statement rather than looked up per row, which would be an
        N+1 across the page.
        """
        stmt = (
            select(VisibilityData, Target.name)
            .join(Target, Target.id == VisibilityData.target_id)
            .where(
                and_(
                    VisibilityData.night_date == night_date,
                    VisibilityData.site_id == site_id,
                    VisibilityData.remaining_minutes >= min_remaining_minutes,
                )
            )
            .order_by(
                VisibilityData.remaining_minutes.desc(),
                VisibilityData.observation_id,
            )
            .offset(offset)
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return [(row[0], row[1]) for row in result.all()]

    async def count_visible_on_night(
        self,
        night_date: date,
        site_id: int,
        min_remaining_minutes: int = 0,
    ) -> int:
        """How many rows ``get_visible_on_night`` would return unpaginated.

        No join and no LIMIT: the page total is just a number, and either would
        make it slower or wrong.
        """
        stmt = select(func.count()).select_from(VisibilityData).where(
            and_(
                VisibilityData.night_date == night_date,
                VisibilityData.site_id == site_id,
                VisibilityData.remaining_minutes >= min_remaining_minutes,
            )
        )
        result = await self.session.execute(stmt)
        return int(result.scalar_one() or 0)

    async def get_stored_observation_ids_on_night(
        self,
        night_date: date,
        site_id: int | None = None,
    ) -> set[str]:
        """Observation ids with visibility stored for a night.

        Ids only — the coverage check never needs the JSONB payloads, and
        dragging them back would dominate the query cost.
        """
        conditions = [VisibilityData.night_date == night_date]
        if site_id is not None:
            conditions.append(VisibilityData.site_id == site_id)
        stmt = select(VisibilityData.observation_id).where(
            and_(*conditions)
        ).distinct()
        result = await self.session.execute(stmt)
        return {row[0] for row in result.all()}

    async def upsert(
        self,
        observation_id: str,
        target_id: int,
        site_id: int,
        night_date: date,
        remaining_minutes: int,
        visible_ranges: list,
        constraints: dict,
    ) -> VisibilityData:
        """Insert or update visibility data."""
        stmt = select(VisibilityData).where(
            and_(
                VisibilityData.observation_id == observation_id,
                VisibilityData.target_id == target_id,
                VisibilityData.site_id == site_id,
                VisibilityData.night_date == night_date,
            )
        )
        result = await self.session.execute(stmt)
        existing = result.scalar_one_or_none()
        
        if existing:
            existing.remaining_minutes = remaining_minutes
            existing.visible_ranges = visible_ranges
            existing.constraints = constraints
            await self.session.flush()
            return existing
        
        data = VisibilityData(
            observation_id=observation_id,
            target_id=target_id,
            site_id=site_id,
            night_date=night_date,
            remaining_minutes=remaining_minutes,
            visible_ranges=visible_ranges,
            constraints=constraints,
        )
        self.session.add(data)
        await self.session.flush()
        return data

    async def bulk_upsert(self, rows: list[dict]) -> int:
        """
        Insert or update many visibility rows in a few round trips.

        Every row dict must have exactly the keys observation_id, target_id,
        site_id, night_date, remaining_minutes, visible_ranges, constraints
        (identical key sets across dicts, or SQLAlchemy compiles the column
        list from the first dict only). The statement has no RETURNING, so
        asyncpg pipelines each chunk as a single executemany round trip
        instead of one SELECT + one write per row. asyncpg reports no
        rowcount for executemany, so the returned count is rows submitted.
        """
        if not rows:
            return 0

        insert_stmt = pg_insert(VisibilityData)
        stmt = insert_stmt.on_conflict_do_update(
            constraint="uq_visibility_observation_night",
            set_={
                "remaining_minutes": insert_stmt.excluded.remaining_minutes,
                "visible_ranges": insert_stmt.excluded.visible_ranges,
                "constraints": insert_stmt.excluded.constraints,
                "computed_at": func.now(),
            },
        )
        for start in range(0, len(rows), BULK_UPSERT_CHUNK):
            await self.session.execute(stmt, rows[start:start + BULK_UPSERT_CHUNK])
        return len(rows)

    async def delete_by_observation(
        self,
        observation_id: str,
    ) -> int:
        """Delete all visibility data for an observation."""
        stmt = delete(VisibilityData).where(
            VisibilityData.observation_id == observation_id
        )
        result = await self.session.execute(stmt)
        return result.rowcount
