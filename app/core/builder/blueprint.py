# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
from typing import FrozenSet, Union, List, Iterable

from astropy.time import TimeDelta
import astropy.units as u

from lucupy.minimodel.observation import ObservationClass
from lucupy.minimodel.program import ProgramTypes
from lucupy.minimodel.semester import Semester, SemesterHalf
from lucupy.minimodel.site import Site, ALL_SITES

from app.config import config, ConfigurationError
from app.core.components.optimizer.dummy import DummyOptimizer

class Blueprint:
    """Base class for Blueprint
    """
    pass

class CollectorBlueprint(Blueprint):
    """Blueprint for the Collector.
    This is based on the configuration in config.yml.
    """
    def __init__(self,
                 semesters: str,
                 obs_class: str,
                 prg_type: str,
                 sites: str,
                 time_slot_length: float) -> None:
        self.semesters: FrozenSet[Semester] =  frozenset(map(CollectorBlueprint._parse_semesters, semesters))
        self.obs_classes: FrozenSet[ObservationClass] = frozenset(map(CollectorBlueprint._parse_obs_class, obs_class))
        self.program_types: FrozenSet[ProgramTypes] = frozenset(map(CollectorBlueprint._parse_prg_types,prg_type))
        self.sites: FrozenSet[Site] = CollectorBlueprint._parse_sites(sites)
        self.time_slot_length: TimeDelta = TimeDelta(time_slot_length * u.min)

    @staticmethod
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
        except KeyError:
            raise ConfigurationError('Semester Half', half)

        try:
            return Semester(int(year), e_half)
        except ValueError:
            raise ConfigurationError('Semester year', year)    

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _parse_sites(sites: Union[str, List[str]]) -> FrozenSet[Site]:
        """Parse Sites in config.yml

        Args:
            sites (Union[str, List[str]]): Option can be a list of sites or a single one

        Returns:
            FrozenSet[Site]: a frozen site that contains lucupy Site enums 
                corresponding to each site.
        """

        def parse_specific_site(site: str):
            if site == 'GS':
                return Site.GS
            elif site == 'GN':
                return Site.GN
            else:
                raise ConfigurationError('Missing site', site)
        if sites == 'ALL_SITES':
        # In case of ALL_SITES option, return lucupy alias for the set of all Site enums
            return ALL_SITES 

        if isinstance(sites, list):
            return frozenset(map(parse_specific_site, sites))
        else:
            # Single site case
            return frozenset([parse_specific_site(sites)])

    def __iter__(self):
        return iter((self.time_slot_length,
                     self.sites,
                     self.semesters, 
                     self.program_types,
                     self.obs_classes))

class OptimizerBlueprint(Blueprint):
    """Blueprint for the Selector.
    This is based on the configuration in config.yml.
    """
    
    def __init__(self, algorithm: str) -> None:
        self.algorithm = OptimizerBlueprint._parse_optimizer(algorithm)
    
    @staticmethod
    def _parse_optimizer(algorithm_name: str):
        # TODO: Enums are needed but for now is just Dummy
        # TODO: When GMax is ready we can expand
        if algorithm_name in 'Dummy':
            return DummyOptimizer()
        else:
            raise ConfigurationError('Optimizer', config.optimizer.name)
    def __iter__(self):
        return iter((self.algorithm))

class Blueprints:
    collector: CollectorBlueprint = CollectorBlueprint(config.collector.semesters,
                                                       config.collector.observation_classes,
                                                       config.collector.program_types,
                                                       config.collector.sites,
                                                       config.collector.time_slot_length)
    optimizer: OptimizerBlueprint = OptimizerBlueprint(config.optimizer.name)
