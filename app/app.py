# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from fastapi import FastAPI
from app.graphql import graphql_server

app = FastAPI()
app.add_route('/graphql', graphql_server)
app.add_websocket_route('/graphql', graphql_server)
