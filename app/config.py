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

    if half == 'A':
        e_half = SemesterHalf.A
    elif half == 'B':
        e_half = SemesterHalf.B
    else:
        ConfigurationError('Semester Half', half)

    try:
        return Semester(int(year), e_half)
    except:
        raise ConfigurationError('Semester year', year)    

def _parse_obs_class(obs_class: str) -> ObservationClass:
    """Parse Observation class from config.yml

    Args:
        obs_class (str): _description_

    Raises:
        ConfigurationError: _description_

    Returns:
        _type_: _description_
    """

    if obs_class == 'SCIENCE':
        return ObservationClass.SCIENCE
    elif obs_class == 'PROGCAL':
        return ObservationClass.PROGCAL
    elif obs_class == 'PARTNERCAL':
        return ObservationClass.PARTNERCAL
    elif obs_class == 'ACQ':
        return ObservationClass.ACQ
    elif obs_class == 'ACQCAL':
        return ObservationClass.ACQCAL
    elif obs_class == 'DAYCAL':
        return ObservationClass.DAYCAL
    else:
        raise ConfigurationError('Observation class', obs_class)

def _parse_prg_types(prg_type: str) -> ProgramTypes:
    """Parse Program type from config.yml

    Args:
        prg_type (str): Value from 

    Raises:
        ConfigurationError: _description_

    Returns:
        _type_: _description_
    """
    if prg_type == 'C':
        return ProgramTypes.C
    elif prg_type == 'CAL':
        return ProgramTypes.CAL
    elif prg_type == 'DD':
        return ProgramTypes.DD
    elif prg_type == 'DS':
        return ProgramTypes.DS
    elif prg_type== 'ENG':
        return ProgramTypes.ENG
    elif prg_type == 'FT':
        return ProgramTypes.FT
    elif prg_type == 'LP':
        return ProgramTypes.LP
    elif prg_type == 'Q':
        return ProgramTypes.Q
    elif prg_type == 'SV':
        return ProgramTypes.SV
    else:
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
        if site == 'GS':
            return Site.GS
        elif site == 'GN':
            return Site.GN
        else:
            raise ConfigurationError('Missing site', site)
    if sites == 'ALL_SITES':
    # In case of ALL_SITES option, return lucupy alias for the set of all Site enums
        return ALL_SITES 

    if isinstance(config.scheduler.sites, list):
        return frozenset(list(map(parse_site_specfic, config.scheduler.sites)))
    else:
        # Single site case
        return frozenset([parse_site_specfic(sites)])


@dataclass
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


