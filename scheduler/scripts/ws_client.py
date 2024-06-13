import asyncio
import json
from datetime import datetime
from websockets.exceptions import ConnectionClosed
import websockets
import time

# Define the data to be sent
datesState = [
    datetime(2018, 9, 1, ),  # Example start date
    datetime(2018, 9, 3,)   # Example end date
]
siteState = ["GN", "GS"]  # Example site state
numNight = 1  # Example number of nights to schedule
thesis = 1.1
power = 2
metPower = 1.0
visPower = 1.0
whaPower = 1.0

payload = {
    "startTime": datesState[0].isoformat().split(".")[0].replace("T", " "),
    "endTime": datesState[1].isoformat().split(".")[0].replace("T", " "),
    "sites": siteState,
    "schedulerMode": "VALIDATION",
    "semesterVisibility": "True",
    "numNightsToSchedule": numNight,
    "rankerParameters": {
        "thesisFactor": thesis,
        "power": power,
        "metPower": metPower,
        "visPower": visPower,
        "whaPower": whaPower,
    }
}


async def connect():
    client_id = int(time.time())
    uri = f"ws://gpp-schedule-staging.herokuapp.com/ws/{client_id}"  # Replace with your WebSocket server URI
    async with websockets.connect(uri, ping_interval=None) as websocket:
        # Send a message
        try:
            while True:
                await websocket.send(json.dumps(payload))

                # Receive a message
                response = await websocket.recv()
                print(f"Received: {response}")
        except (ConnectionError, ConnectionClosed) as e:
            print(f'error!: {e}')
        finally:
            await websocket.close()
            print("WebSocket connection closed")

asyncio.run(connect())
