from fastapi import FastAPI
from app.graphql import graphql_server
from app.routers import router

app = FastAPI()
# app.include_router(graphql_app, prefix='/graphql')
app.include_router(router)
app.add_route('/graphql', graphql_server)
app.add_websocket_route('/graphql', graphql_server)
