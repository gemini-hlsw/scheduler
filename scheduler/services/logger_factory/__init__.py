# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import logging


def create_logger(name: str, level: int = logging.WARNING) -> logging.Logger:
    """
    Create a Logger to be used for logging, configured with the given name, which should be the class
    or module from where the logging is being done.

    The format of the output will be: time - name - level - message

    Args:
        name (str): the name of the module or class in which the logger will be used
        level (int): the logging level; values should be taken from the Python logging module

    Returns:
        a configured Logger that can be used for logging
    """
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
