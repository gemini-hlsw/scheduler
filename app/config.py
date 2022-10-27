# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass
import os
from astropy.time import TimeDelta
import astropy.units as u
from omegaconf import OmegaConf

from definitions import ROOT_DIR
from typing import FrozenSet, Union, List

from lucupy.minimodel.observation import ObservationClass
from lucupy.minimodel.program import ProgramTypes
from lucupy.minimodel.semester import Semester, SemesterHalf
from lucupy.minimodel.site import Site, ALL_SITES


class ConfigurationError(Exception):
    """Exception raised for errors in parsing configuration.
    
    Attributes:
         config_type (str): Type of configuration that was being parse at the moment of the error.
         value (str): Value that causes the error.
    """
    def __init__(self, config_type: str, value: str):
        super().__init__(f'Configuration error: {config_type} {value} is invalid.')

def _parse_semesters(semester: str) -> Semester:
    """Parse semesters to schedule from config.yml

    Args:
        semester (str): _description_

    Raises:
        ConfigurationError: _description_

    Returns:
        Semester: _description_
    """

    year, half = semester[:-1], semester[-1]

    try:
        e_half = SemesterHalf[half]
    except:
        ConfigurationError('Semester Half', half)

    try:
        return Semester[int(year), e_half]
    except:
        raise ConfigurationError('Semester year', year)    

def _parse_obs_class(obs_class: str) -> ObservationClass:
    """Parse Observation class from config.yml

    Args:
        obs_class (str): Option from config

    Raises:
        ConfigurationError: If class not in the Enum lookup

    Returns:
        ObservationClass: Corresponding enum value
    """
    try:
        return ObservationClass[obs_class]
    except:
        raise ConfigurationError('Observation class', obs_class)
     
def _parse_prg_types(prg_type: str) -> ProgramTypes:
    """Parse Program type from config.yml

    Args:
        prg_type (str): Option on config

    Raises:
        ConfigurationError: If type not in the Enum lookup

    Returns:
        ProgramTypes: Enum value
    """
    try:
        return ProgramTypes[prg_type]
    except:
        raise ConfigurationError('Program type', prg_type)   

def _parse_sites(sites: Union[str, List[str]]) -> FrozenSet[Site]:
    """Parse Sites in config.yml

    Args:
        sites (Union[str, List[str]]): Option can be a list of sites or a single one

    Returns:
        FrozenSet[Site]: a frozen site that contains lucupy Site enums 
            corresponding to each site.
    """

    def parse_site_specfic(site: str):
        try:
            return Site[site]
        except:
            raise ConfigurationError('Missing site', site)
    
    if sites == 'ALL_SITES':
    # In case of ALL_SITES option, return lucupy alias for the set of all Site enums
        return ALL_SITES 

    if isinstance(config.scheduler.sites, list):
        return frozenset(list(map(parse_site_specfic, config.scheduler.sites)))
    else:
        # Single site case
        return frozenset([parse_site_specfic(sites)])


@dataclass(frozen=True)
class CollectorConfig:
    semesters: FrozenSet[Semester]
    obs_classes: FrozenSet[ObservationClass]
    program_types: FrozenSet[ProgramTypes]
    sites: FrozenSet[Site]
    time_slot_length: TimeDelta


path = os.path.join(ROOT_DIR, 'config.yaml')
config = OmegaConf.load(path)

config_collector = CollectorConfig(frozenset(map(_parse_semesters, config.collector.semesters)), 
                   frozenset(map(_parse_obs_class, config.collector.observation_classes)),
                   frozenset(map(_parse_prg_types, config.collector.program_types)),
                   _parse_sites(config.collector.sites),
                   TimeDelta(config.collector.time_slot_length * u.min)
                   )


