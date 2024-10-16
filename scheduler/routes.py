# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
import asyncio
import json

from fastapi.responses import JSONResponse
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
from fastapi.responses import HTMLResponse
from starlette.requests import Request

from scheduler.app import app
from scheduler.connection_manager import ConnectionManager
from scheduler.engine import Engine
from scheduler.engine.params import SchedulerParameters
from scheduler.services import logger_factory
from scheduler.graphql_mid.types import SNightTimelines


_logger = logger_factory.create_logger(__name__)
manager = ConnectionManager()


# Root API
@app.get("/", include_in_schema=False)
def root() -> JSONResponse:
    return JSONResponse(status_code=200,
                        content={
                            "message": "Welcome to Server"})


def worker(data: dict) -> dict:
    params = SchedulerParameters.from_json(data)
    engine = Engine(params)
    plan_summary, timelines = engine.run()
    return {"plan_summary": plan_summary, "timelines": timelines.to_json()}


async def keep_alive(websocket: WebSocket) -> None:
    """Sends a ping to the websocket to avoid idle timeouts."""
    while True:
        try:
            if websocket.application_state == WebSocketState.CONNECTED:
                await manager.send({"type": "ping", "payload": {}}, websocket)
            await asyncio.sleep(30)  # Ping every 30 seconds
        except Exception as e:
            _logger.warning(f"Keep-alive error: {e}")
            break


async def websocket_handler(websocket: WebSocket) -> None:
    """Handles the task that allows to run the Scheduler engine"""
    while True:
        data = await websocket.receive_json()
        if data:
            task = asyncio.to_thread(worker, data)
            await manager.send({"type": "update", "payload": {"message": "Processing plans..."}}, websocket)
            result = await task
            await manager.send({"type": "plans", "payload": result}, websocket)
        else:
            raise ValueError('Missing parameters to create schedule')


@app.websocket("/ws/{client_id}")
async def schedule_websocket(websocket: WebSocket, client_id: int):

    await manager.connect(websocket)

    keep_alive_task = asyncio.create_task(keep_alive(websocket))
    handler_task = asyncio.create_task(websocket_handler(websocket))

    try:
        await asyncio.gather(keep_alive_task, handler_task)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        _logger.warning(f"An error occurred in websocket {client_id}: {e}")
    finally:
        keep_alive_task.cancel()
        handler_task.cancel()
        await keep_alive_task
        await handler_task


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
            <label for="startTime">Start:</label>
            <input type="text" id="startTime" value="2018-10-01 08:00:00" autocomplete="off"/><br/>
            <label for="endTime">End:</label>
            <input type="text" id="endTime" value="2018-10-03 08:00:00" autocomplete="off"/><br/>
            <label for="sites">Sites:</label>
            <input type="text" id="sites" value="GN" autocomplete="off"/><br/>
            <label for="mode">Mode:</label>
            <input type="text" id="mode" value="VALIDATION" autocomplete="off"/><br/>
            <label for="semesterVisibility">semester Visibility:</label>
            <input type="text" id="semesterVisibility" value="True" autocomplete="off"/><br/>
            <label for="numNightsToSchedule"> num nights to schedule:</label>
            <input type="text" id="numNightsToSchedule" value="3" autocomplete="off"/><br/>
            <label for="thesisFactor">thesis_actor:</label>
            <input type="number" id="thesisFactor" value="1.1" autocomplete="off"/><br/>
            <label for="power">power:</label>
            <input type="number" id="power" value="2" autocomplete="off"/><br/>
            <label for="metPower">met_power:</label>
            <input type="number" id="metPower" value="1.0" autocomplete="off"/><br/>
            <label for="visPower">vis_power:</label>
            <input type="number" id="visPower" value= "1.0" autocomplete="off"/><br/>
            <label for="whaPower">wha_power:</label>
            <input type="number" id="whaPower" value= "1.0" autocomplete="off"/><br/>
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
                var content = document.createTextNode(JSON.stringify(JSON.parse(event.data), null, 4))
                message.appendChild(content)
                messages.appendChild(message)
            };
            function sendMessage(event) {

                var input1 = document.getElementById("startTime")
                var input2 = document.getElementById("endTime")
                var input3 = document.getElementById("sites")
                var input4 = document.getElementById("mode")
                var input5 = document.getElementById("semesterVisibility")
                var input6 = document.getElementById("numNightsToSchedule")
                var input7 = document.getElementById("thesisFactor")
                var input8 = document.getElementById("power")
                var input9 = document.getElementById("metPower")
                var input10 = document.getElementById("visPower")
                var input11 = document.getElementById("whaPower")

                var data = {
                    startTime: input1.value,
                    endTime: input2.value,
                    sites: [input3.value ],
                    schedulerMode: input4.value,
                    semesterVisibility: input5.value,
                    numNightsToSchedule: input6.value,
                    rankerParameters: {
                        thesisFactor: input7.value,
                        power: input8.value,
                        metPower: input9.value,
                        visPower: input10.value,
                        whaPower: input11.value,
                    }
                }
                ws.send(JSON.stringify(data));
                event.preventDefault()
            }
        </script>
    </body>
</html>
""")
