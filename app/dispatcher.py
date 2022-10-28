import uvicorn
import os
import logging
from typing import Optional

from app import app
from .config import config
from app.process_manager import ProcessManager
from app.core.scheduler.modes import SchedulerModes
from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties


def dispatch(heroku_port: Optional[int] = None):

    # Setup scheduler mode
    try:
        mode = SchedulerModes[config.mode.upper()]
    except:
        raise ValueError('Mode is Invalid!')

    # Setup lucupy properties 
    ObservatoryProperties.set_properties(GeminiProperties)

    # Start process manager
    if mode is SchedulerModes.OPERATION: 
        manager = ProcessManager(size=1, timeout=config.process_manager.timeout)
    else:
        manager = ProcessManager(size=config.process_manager.size,
                                    timeout=config.process_manager.timeout)

    uvicorn.run(app, host=config.server.host, port= heroku_port if heroku_port else config.server.port)
    