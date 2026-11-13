# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from typing import Any, Optional, Union

import strawberry  # noqa
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.types import Receive, Scope, Send
from starlette.websockets import WebSocket
from strawberry.asgi import GraphQL  # noqa
from strawberry.exceptions import ConnectionRejectionError

from scheduler.auth import AuthError, AuthSettings, authenticate, auth_settings
from scheduler.services.logger_factory import create_logger

from .schema import Query, Subscription, Mutation

_logger = create_logger(__name__)

_USER_SCOPE_KEY = "scheduler.sso_user"


class AuthenticatedGraphQL(GraphQL):
    """GraphQL over HTTP and websockets, gated on a GPP SSO token.

    HTTP requests carry it in the Authorization header and are refused with
    401/403 before anything executes. Websockets carry it in the graphql-ws
    ``connection_init`` payload (``{"Authorization": "Bearer …"}``) and are
    closed with 4403 when it is refused. The verified user is in the context
    as ``context["user"]`` (None only when auth is not required).
    """

    def __init__(self, *args: Any, settings: Optional[AuthSettings] = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._settings = settings

    @property
    def settings(self) -> AuthSettings:
        return self._settings or auth_settings()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # HTTP is checked before strawberry sees the request, so a refused one
        # never executes. CORS preflights are answered by the middleware first.
        if scope["type"] == "http":
            request = Request(scope, receive)
            try:
                scope[_USER_SCOPE_KEY] = authenticate(request.headers.get("authorization"), self.settings)
            except AuthError as e:
                headers = {"WWW-Authenticate": "Bearer"} if e.status_code == 401 else None
                await PlainTextResponse(str(e), status_code=e.status_code, headers=headers)(scope, receive, send)
                return
        await super().__call__(scope, receive, send)

    async def get_context(
        self, request: Union[Request, WebSocket], response: Union[Response, WebSocket]
    ) -> dict[str, Any]:
        # A websocket's user is set in on_ws_connect, from its connection_init payload.
        return {"request": request, "response": response, "user": request.scope.get(_USER_SCOPE_KEY)}

    async def on_ws_connect(self, context: dict[str, Any]):
        params = context.get("connection_params") or {}
        authorization = params.get("Authorization") or params.get("authorization")
        try:
            context["user"] = authenticate(authorization, self.settings)
        except AuthError as e:
            _logger.info(f"Websocket refused: {e}")
            raise ConnectionRejectionError({"message": str(e)}) from None
        return await super().on_ws_connect(context)


schema = strawberry.Schema(query=Query, subscription=Subscription, mutation=Mutation)
graphql_server = AuthenticatedGraphQL(schema)
