# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
from fastapi import Depends
from fastapi.responses import JSONResponse
from lucupy.minimodel import Site

from scheduler.app import app
from scheduler.engine.params import build_params_store, BuildParameters
from scheduler.server.process_manager import process_manager


# Root API
@app.get("/", include_in_schema=False)
def root() -> JSONResponse:
    return JSONResponse(status_code=200,
                        content={
                            "message": "Welcome to Server"})

@app.get("/get_operation_id")
def get_operation_id() -> JSONResponse:
    return JSONResponse(
        status_code=200,
        content={
            "message": process_manager.get_operation_id()
    })
