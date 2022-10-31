# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from typing import Optional

import uvicorn
from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties

from app import app
from app.core.scheduler.modes import SchedulerModes
from app.process_manager import ProcessManager
from .config import config


def dispatch(heroku_port: Optional[int] = None):
    # Setup scheduler mode
    try:
        mode = SchedulerModes[config.mode.upper()]
    except KeyError:
        raise ValueError('Mode is Invalid!')

    # Setup lucupy properties 
    ObservatoryProperties.set_properties(GeminiProperties)

    # Start process manager
    if mode is SchedulerModes.OPERATION:
        manager = ProcessManager(size=1, timeout=config.process_manager.timeout)
    else:
        manager = ProcessManager(size=config.process_manager.size,
                                 timeout=config.process_manager.timeout)

    uvicorn.run(app, host=config.server.host, port=heroku_port if heroku_port else config.server.port)
