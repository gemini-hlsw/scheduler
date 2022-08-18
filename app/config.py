from dotenv import load_dotenv
import os
from omegaconf import OmegaConf
from definitions import ROOT_DIR

load_dotenv()

path = os.path.join(ROOT_DIR, 'config.yaml')

config = OmegaConf.load(path)
config.graphql.url = os.environ.get("GRAPHQL_URL")  # HACK: to get the url from the env
