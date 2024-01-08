# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import os
import sys
import logging
from enum import IntEnum
from typing import NoReturn

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class LoggingLevels(IntEnum):
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    FATAL = logging.FATAL
    OFF = 100


def _print_usage() -> NoReturn:
    print(f'Usage: {sys.argv[0]} <logging_level>')
    sys.exit(1)


# Default is full logging.
if len(sys.argv) == 1:
    DEFAULT_LOGGING_LEVEL = LoggingLevels.INFO
elif len(sys.argv) == 2:
    try:
        DEFAULT_LOGGING_LEVEL = LoggingLevels[sys.argv[1].upper()]
    except KeyError:
        _print_usage()
else:
    _print_usage()
