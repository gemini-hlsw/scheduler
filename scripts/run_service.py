import logging
import asyncio
import os
from omegaconf import OmegaConf
from app import App
from mock.observe import Observe


# TODO: This script should be the main entry point for the application in the future.

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    
    path = os.path.join(os.getcwd(), '..', 'config.yaml')
    config = OmegaConf.load(path)

    app = App(config)
    asyncio.run(app.run())
    # Observe.start()
    
