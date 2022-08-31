# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import os

from dotenv import load_dotenv

load_dotenv()

ODB_ENDPOINT_URL = os.environ.get("ODB_ENDPOINT_URL")
CORE_ENDPOINT_URL = os.environ.get("CORE_ENDPOINT_URL")
