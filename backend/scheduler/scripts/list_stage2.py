"""List Stage 2 visibility data stored in the sight DB (per-observation summary)."""

import asyncio

from sqlalchemy import func, select

from scheduler.services.sight.database.connection import init_engine, session_scope
from scheduler.services.sight.database.models import Site, Target, VisibilityData


async def main():
    await init_engine()
    async with session_scope() as session:
        stmt = (
            select(
                VisibilityData.observation_id,
                Target.name.label('target_name'),
                Site.name.label('site_name'),
                func.count(VisibilityData.id).label('nights'),
                func.min(VisibilityData.night_date).label('min_date'),
                func.max(VisibilityData.night_date).label('max_date'),
                func.sum(VisibilityData.remaining_minutes).label('total_minutes'),
            )
            .join(Target, VisibilityData.target_id == Target.id)
            .join(Site, VisibilityData.site_id == Site.id)
            .group_by(VisibilityData.observation_id, Target.name, Site.name)
            .order_by(VisibilityData.observation_id)
        )
        rows = (await session.execute(stmt)).all()

        if not rows:
            print('No Stage 2 (visibility) data found.')
            return

        total = (await session.execute(select(func.count(VisibilityData.id)))).scalar()
        print(f'Total Stage 2 records: {total}')
        print(f'Unique observation/target/site combinations: {len(rows)}')
        print()
        print(f"{'Observation':<25} {'Target':<25} {'Site':<12} {'Nights':<8} {'Total Min':<12} Date Range")
        print('-' * 110)
        for row in rows:
            print(
                f'{row.observation_id:<25} {row.target_name:<25} {row.site_name:<12} '
                f'{row.nights:<8} {row.total_minutes:<12} {row.min_date} to {row.max_date}'
            )


if __name__ == '__main__':
    asyncio.run(main())
