from dotenv import load_dotenv
import os

load_dotenv()

ODB_ENDPOINT_URL = os.environ.get("ODB_ENDPOINT_URL") 
CORE_ENDPOINT_URL = os.environ.get("CORE_ENDPOINT_URL")
