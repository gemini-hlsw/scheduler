# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
import os
import uvicorn

from lucupy.observatory.gemini.geminiproperties import GeminiProperties  # isort: skip
from lucupy.observatory.abstract import ObservatoryProperties  # isort: skip

from scheduler.app import app
from scheduler.config import config

from scheduler.server.process_manager import ProcessManager
import asyncio


heroku_port = os.environ.get("PORT")
heroku_port = int(heroku_port) if isinstance(heroku_port, str) else heroku_port


async def main():
    # Setup lucupy properties
    # TODO: This should be dynamic but since we are just working with Gemini right now
    #       should not be an issue.
    ObservatoryProperties.set_properties(GeminiProperties)
    uvicorn_config = uvicorn.Config(app,
                host=config.server.host,
                port=heroku_port if heroku_port else config.server.port,
                ws_ping_interval=50)

    server = uvicorn.Server(uvicorn_config)

    # Start the server and the process manager

    await server.serve(),


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down scheduler server...")
        # asyncio.get_event_loop().stop()
