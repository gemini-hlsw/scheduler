import strawberry
from strawberry.asgi import GraphQL
from .schema import Query, Mutation

schema = strawberry.Schema(query=Query, mutation=Mutation)
graphql_server = GraphQL(schema)
