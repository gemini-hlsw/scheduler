# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from scheduler.graphql_mid.server import graphql_server

app = FastAPI()

origins = [
    "https://schedule-subscriber-staging.herokuapp.com",
    "https://gpp-schedule-ui-staging.herokuapp.com",
    "http://scheduler.lucuma.xyz:5173" # dev
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
