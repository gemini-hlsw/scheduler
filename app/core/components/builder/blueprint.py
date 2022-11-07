from dataclasses import dataclass
from typing import FrozenSet, Union, List, ClassVar

from astropy.time import TimeDelta
import astropy.units as u

from lucupy.minimodel.observation import ObservationClass
from lucupy.minimodel.program import ProgramTypes
from lucupy.minimodel.semester import Semester, SemesterHalf
from lucupy.minimodel.site import Site, ALL_SITES

from app.config import config, ConfigurationError

class Blueprint:
    pass

class CollectorBlueprint(Blueprint):

    def __init__(self, 
                semesters: str, 
                obs_class: str, 
                prg_type: str, 
                sites: str, 
                time_slot_length: float) -> None:
        self.semesters: FrozenSet[Semester] =  frozenset(map(CollectorBlueprint._parse_semesters, semesters))
        self.obs_classes: FrozenSet[ObservationClass] = frozenset(map(CollectorBlueprint._parse_obs_class, obs_class))
        self.program_types: FrozenSet[ProgramTypes] = frozenset(map(CollectorBlueprint._parse_prg_types,prg_type))
        self.sites: FrozenSet[Site] = frozenset(map(CollectorBlueprint._parse_sites, sites))
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

    def __iter__(self):
        return (self.semesters, self.obs_classes, self.program_types, self.sites, self.time_slot_length)


class Blueprints:
    collector: ClassVar[CollectorBlueprint] = CollectorBlueprint(config.collector.semesters,
                                                       config.collector.obs_clasess,
                                                       config.collector.program_types,
                                                       config.collector.sites)