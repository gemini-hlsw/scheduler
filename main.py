# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import os

import uvicorn
from fastapi.responses import JSONResponse
from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties

from app import app
from app.config import config

heroku_port = os.environ.get("PORT")


# Root API
@app.get("/", include_in_schema=False)
def root() -> JSONResponse:
    return JSONResponse(status_code=200,
                        content={
                            "message": "Welcome to Server"})


def main():
    # Setup lucupy properties
    # TODO: This should be dynamic but since we are just working with Gemini right now
    #       should not be an issue.
    ObservatoryProperties.set_properties(GeminiProperties)
    uvicorn.run(app, host=config.server.host, port=heroku_port if heroku_port else config.server.port)


if __name__ == "__main__":
    main()
