# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
import asyncio
import os

import uvicorn
from astropy.time import Time
from fastapi.responses import JSONResponse
from fastapi import WebSocket, WebSocketDisconnect
from lucupy.minimodel import ALL_SITES
from lucupy.observatory.gemini.geminiproperties import GeminiProperties  # isort: skip
from lucupy.observatory.abstract import ObservatoryProperties  # isort: skip
from fastapi.responses import HTMLResponse
from starlette.requests import Request

from scheduler.app import app
from scheduler.config import config
from scheduler.connection_manager import ConnectionManager
from scheduler.core.builder.modes import SchedulerModes
from scheduler.core.components.ranker import RankerParameters
from scheduler.core.service import Service

heroku_port = os.environ.get("PORT")
manager = ConnectionManager()

# Root API
@app.get("/", include_in_schema=False)
def root() -> JSONResponse:
    return JSONResponse(status_code=200,
                        content={
                            "message": "Welcome to Server"})


async def ping_server(websocket):
    while True:
        await websocket.ping()
        await asyncio.sleep(5)


@app.websocket("/ws/{client_id}")
async def schedule_websocket(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data:
                for n in Service().generate(SchedulerModes.VALIDATION,
                                            Time("2018-10-01 08:00:00", format='iso', scale='utc'),
                                            Time("2018-10-06 08:00:00", format='iso', scale='utc'),
                                            ALL_SITES,
                                            RankerParameters(),
                                            True,
                                            4,
                                            None):
                    await manager.send(n, websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/websocket-client")
async def websocket_client(request: Request):
    return HTMLResponse("""
<!DOCTYPE html>
<html>
    <head>
        <title>Schedule</title>
    </head>
    <body>
        <h1>WebSocket Plans</h1>
        <h2>Your ID: <span id="ws-id"></span></h2>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off"/>
            <button>Run</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            var client_id = Date.now()
            document.querySelector("#ws-id").textContent = client_id;
            var ws = new WebSocket(`ws://localhost:8000/ws/${client_id}`);
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(JSON.stringify(event.data, null, 2))
                message.appendChild(content)
                messages.appendChild(message)
            };
            function sendMessage(event) {
                var input = document.getElementById("messageText")
                ws.send(input.value)
                input.value = ''
                event.preventDefault()
            }
        </script>
    </body>
</html>
""")


def main():
    # Setup lucupy properties
    # TODO: This should be dynamic but since we are just working with Gemini right now
    #       should not be an issue.
    ObservatoryProperties.set_properties(GeminiProperties)
    uvicorn.run(app,
                host=config.server.host,
                port=heroku_port if heroku_port else config.server.port,
                ws_ping_interval=50)


if __name__ == "__main__":
    main()
