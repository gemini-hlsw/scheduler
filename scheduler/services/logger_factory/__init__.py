# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import logging
from typing import Optional

from definitions import DEFAULT_LOGGING_LEVEL


def create_logger(name: str, level: Optional[int] = DEFAULT_LOGGING_LEVEL) -> logging.Logger:
    """
    Create a Logger to be used for logging, configured with the given name, which should be the class
    or module from where the logging is being done.

    The format of the output will be: time - name - level - message

    Args:
        name (str): the name of the module or class in which the logger will be used
        level (int): the logging level; values should be taken from the Python logging module or None to disable logging

    Returns:
        a configured Logger that can be used for logging
    """
    if level is None:
        level = 100

    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
