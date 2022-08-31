# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import strawberry
from strawberry.asgi import GraphQL

from .schema import Query, Mutation

schema = strawberry.Schema(query=Query, mutation=Mutation)
graphql_server = GraphQL(schema)
