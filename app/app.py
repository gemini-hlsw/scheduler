from fastapi import FastAPI
from app.graphql import graphql_server

app = FastAPI()
app.add_route('/graphql', graphql_server)
app.add_websocket_route('/graphql', graphql_server)
