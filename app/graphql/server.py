import strawberry
from strawberry.asgi import GraphQL
from .schema import Query

schema = strawberry.Schema(query=Query)
graphql_server = GraphQL(schema)
