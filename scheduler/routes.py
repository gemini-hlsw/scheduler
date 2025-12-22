# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
from fastapi import Depends
from fastapi.responses import JSONResponse
from scheduler.app import app
from scheduler.engine import SchedulerParameters
from scheduler.engine.params import SharedStateSchedulerParameters, SchedulerParametersV2, get_shared_params


scheduler = SharedStateSchedulerParameters()

# Root API
@app.get("/", include_in_schema=False)
def root() -> JSONResponse:
    return JSONResponse(status_code=200,
                        content={
                            "message": "Welcome to Server"})


@app.post("/scheduler_parameters")
async def set_scheduler_parameters(
        scheduler_parameters: SchedulerParametersV2,
        shared_params: SharedStateSchedulerParameters = Depends(get_shared_params)
):
    await shared_params.set_params(scheduler_parameters)



