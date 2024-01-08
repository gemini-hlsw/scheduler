# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import os
import sys
import logging

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) > 1:
    DEFAULT_LOGGING_LEVEL = None
else:
    DEFAULT_LOGGING_LEVEL = logging.INFO
