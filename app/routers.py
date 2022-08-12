import asyncio
from datetime import datetime
from fastapi import APIRouter
from .graphql.queries import Session
from .config import config
from .scheduler import build_scheduler
from .process_manager import ProcessManager
from .process_manager import TaskType
router = APIRouter()


@router.get("/odb")
async def odb_changes_endpoint():
    s = Session(url=config.graphql.url)
    while True:
        resp = await s.subscribe_all()
        if resp:
            print(resp)
            # Run new Schedule
            mode = TaskType.STANDARD
            scheduler = build_scheduler()
            ProcessManager().add_task(datetime.now(), scheduler, mode)
        
        await asyncio.sleep(10)
