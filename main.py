# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import uvicorn
from fastapi.responses import JSONResponse

from app import app
from app.config import config
from app.process_manager import ProcessManager


# Root API
@app.get("/", include_in_schema=False)
def root() -> JSONResponse:
    return JSONResponse(status_code=200,
                        content={
                            "message": "Welcome to Server"})


if __name__ == "__main__":
    manager = ProcessManager(size=config.process_manager.size,
                             timeout=config.process_manager.timeout)
    uvicorn.run(app, host=config.server.host, port=config.server.port)
