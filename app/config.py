from dotenv import load_dotenv
import os
from omegaconf import OmegaConf

load_dotenv()

path = os.path.join(os.getcwd(), '.', 'config.yaml')

config = OmegaConf.load(path)
config.graphql.url = os.environ.get("GRAPHQL_URL")  # HACK: to get the url from the env
