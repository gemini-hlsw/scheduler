import logging
import asyncio
import os
from omegaconf import OmegaConf
from app import App
from dotenv import load_dotenv
from mock.observe import Observe


# TODO: This script should be the main entry point for the application in the future.

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    load_dotenv()

    path = os.path.join(os.getcwd(), '..', 'config.yaml')

    config = OmegaConf.load(path)
    config.graphql.url = os.environ.get("GRAPHQL_URL")  # HACK: to get the url from the env

    app = App(config)
    asyncio.run(app.run())
    # Observe.start()
