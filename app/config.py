# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import os
from dataclasses import dataclass
from typing import FrozenSet, Union, List

import astropy.units as u
from astropy.time import TimeDelta
from lucupy.minimodel.observation import ObservationClass
from lucupy.minimodel.program import ProgramTypes
from lucupy.minimodel.semester import Semester, SemesterHalf
from lucupy.minimodel.site import Site, ALL_SITES
from omegaconf import OmegaConf

from definitions import ROOT_DIR


class ConfigurationError(Exception):
    """Exception raised for errors in parsing configuration.
    
    Attributes:
         config_type (str): Type of configuration that was being parsed at the moment of the error.
         value (str): Value that causes the error.
    """

    def __init__(self, config_type: str, value: str):
        super().__init__(f'Configuration error: {config_type} {value} is invalid.')

path = os.path.join(ROOT_DIR, 'config.yaml')
config = OmegaConf.load(path)
