import uvicorn
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
from omegaconf import OmegaConf
from app.graphql.schema import PlanManager
from app.process_manager import ProcessManager, TaskType
from app.scheduler import Scheduler
from app import app
from astropy.time import Time
from datetime import datetime
import asyncio

plan_manager = PlanManager()


async def new_schedule(manager, scheduler):
    manager.add_task(datetime.now(), scheduler, TaskType.STANDARD)
    await asyncio.sleep(10)

# session = Session(config.graphql.url)

# Root API
@app.get("/", include_in_schema=False)
def root() -> JSONResponse:

    return JSONResponse(status_code=200,
                        content={
                            "message": "Welcome to Server"})


@app.get("/new", include_in_schema=False)
def new() -> JSONResponse:
    start = Time("2018-10-01 08:00:00", format='iso', scale='utc')
    end = Time("2018-10-03 08:00:00", format='iso', scale='utc')
    scheduler = Scheduler(config, start, end, plan_manager)
    asyncio.run(new_schedule(manager, scheduler))
    return JSONResponse(status_code=200,
                        content={ "message": f"New schedule created: {plan_manager.get_plans()}"})

if __name__ == "__main__":
    load_dotenv()

    path = os.path.join(os.getcwd(), '.', 'config.yaml')

    config = OmegaConf.load(path)
    config.graphql.url = os.environ.get("GRAPHQL_URL")  # HACK: to get the url from the env

    manager = ProcessManager(size=config.process_manager.size,
                             timeout=config.process_manager.timeout)
    

    # Run random schedule by process manager
    #  mode = TaskType.STANDARD

    #print(plan_manager.get_plans())
    uvicorn.run(app, host='127.0.0.1', port=8000)
