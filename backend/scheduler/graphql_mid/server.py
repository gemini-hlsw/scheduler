# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import strawberry # noqa
from strawberry.asgi import GraphQL # noqa

from .schema import Query, Subscription, Mutation

schema = strawberry.Schema(query=Query, subscription=Subscription, mutation=Mutation)
graphql_server = GraphQL(schema)
