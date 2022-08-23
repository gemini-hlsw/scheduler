from dotenv import load_dotenv
import os
from omegaconf import OmegaConf
from definitions import ROOT_DIR

path = os.path.join(ROOT_DIR, 'config.yaml')
config = OmegaConf.load(path)

