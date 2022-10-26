import uvicorn
import os
import logging

from app import app
from .config import config
from app.process_manager import ProcessManager

from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties


def dispatch(heroku_port:int = None):

    if config.mode.upper() not in ['VALIDATION', 'SIMULATION', 'OPERATION']:
        raise ValueError('Mode is Invalid!')
    logging.info(f'Scheduler staring on {config.mode} mode')

    # Setup lucupy properties 
    ObservatoryProperties.set_properties(GeminiProperties)

    # Start process manager
    if config.mode == 'Operation': 
        manager = ProcessManager(size=1, timeout=config.process_manager.timeout)
    else:
        manager = ProcessManager(size=config.process_manager.size,
                                    timeout=config.process_manager.timeout)

    uvicorn.run(app, host=config.server.host, port= heroku_port if heroku_port else config.server.port)
    