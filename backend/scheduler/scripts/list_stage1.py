"""List Stage 1 data stored in the sight DB (per-target, per-site night counts)."""

import asyncio

from sqlalchemy import func, select

from scheduler.services.sight.database.connection import init_db_engine, session_scope
from scheduler.services.sight.database.models import Site, Target, TargetNightData


async def main():
    await init_db_engine()
    async with session_scope() as session:
        stmt = (
            select(
                Target.name,
                Site.name.label('site_name'),
                func.count(TargetNightData.id).label('nights'),
                func.min(TargetNightData.night_date).label('min_date'),
                func.max(TargetNightData.night_date).label('max_date'),
            )
            .join(Target, TargetNightData.target_id == Target.id)
            .join(Site, TargetNightData.site_id == Site.id)
            .group_by(Target.name, Site.name)
            .order_by(Target.name, Site.name)
        )
        rows = (await session.execute(stmt)).all()

        if not rows:
            print('No Stage 1 data found.')
            return

        total = (await session.execute(select(func.count(TargetNightData.id)))).scalar()
        print(f'Total Stage 1 records: {total}')
        print(f'Unique target/site combinations: {len(rows)}')
        print()
        print(f"{'Target':<30} {'Site':<15} {'Nights':<10} Date Range")
        print('-' * 80)
        for row in rows:
            print(f'{row.name:<30} {row.site_name:<15} {row.nights:<10} {row.min_date} to {row.max_date}')


if __name__ == '__main__':
    asyncio.run(main())
