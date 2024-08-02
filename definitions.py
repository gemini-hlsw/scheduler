# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import os
import sys
import logging
from enum import IntEnum

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class LoggingLevels(IntEnum):
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    FATAL = logging.FATAL
    OFF = 100


# Default logging from LOG_LEVEL env variable otherwise INFO
try:
    LOG_LEVEL = os.environ['LOG_LEVEL'].upper()
except KeyError:
    LOG_LEVEL = 'INFO'

DEFAULT_LOGGING_LEVEL = LoggingLevels[LOG_LEVEL]

# If LOG_LEVEL is provided in the command line, use that
if len(sys.argv) == 2:
    try:
        DEFAULT_LOGGING_LEVEL = LoggingLevels[sys.argv[1].upper()]
    except KeyError:
        print(f"Invalid log level: {sys.argv[1]}")

