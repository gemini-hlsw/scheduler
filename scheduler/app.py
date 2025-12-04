# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
import asyncio

from os import environ
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from scheduler.graphql_mid.server import graphql_server
from scheduler.services.visibility import visibility_calculator
from scheduler.services.logger_factory import create_logger
from scheduler.core.builder.modes import is_validation

_logger = create_logger(__name__, with_id=False)

_logger.info(f"Running scheduler server version {environ['APP_VERSION']}")

async def lifespan(app: FastAPI):
    if is_validation:
        await visibility_calculator.calculate()
    yield

app = FastAPI(lifespan=lifespan)

origins = [
    "https://schedule-staging.gemini.edu",
    "https://schedule.gemini.edu",
    "https://schedule-subscriber-staging.herokuapp.com",
    "https://gpp-schedule-ui-staging.herokuapp.com",
    "http://scheduler.lucuma.xyz:5173",
    "http://localhost",
    "http://localhost:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_route('/graphql', graphql_server)
app.add_websocket_route('/graphql', graphql_server)


# Import the routes after creating the FastAPI instance
from scheduler import routes
