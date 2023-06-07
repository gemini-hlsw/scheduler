# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
from typing import Any, FrozenSet, List, Type, Union
from enum import Enum

from astropy.time import TimeDelta
import astropy.units as u

from lucupy.minimodel.observation import ObservationClass
from lucupy.minimodel.program import ProgramTypes
from lucupy.minimodel.semester import Semester, SemesterHalf
from lucupy.minimodel.site import Site, ALL_SITES

from scheduler.config import config, ConfigurationError
from scheduler.core.components.optimizer.dummy import DummyOptimizer
from scheduler.core.components.optimizer.greedymax import GreedyMaxOptimizer


def parse_configuration(enum_class: Type[Enum], value: str) -> Any:
    """General parser for config.yml

    Args:
        enum_class (Type): Enum corresponding to the setting to parse
        value     (str): Value on config.yml

    Raises:
        ConfigurationError: General configuration error in case the string

    Returns:
        Any: The selection from the Enum.
    """
    try:
        return enum_class[value]
    except TypeError:
        raise ConfigurationError('Enum type', value)
    except ValueError:
        raise ConfigurationError(enum_class.__name__, value)


class Blueprint:
    """Base class for Blueprint
    """
    pass


class CollectorBlueprint(Blueprint):
    """Blueprint for the Collector.
    This is based on the configuration in config.yml.
    """

    def __init__(self,
                 obs_class: List[str],
                 prg_type: List[str],
                 time_slot_length: float) -> None:
        self.obs_classes: FrozenSet[ObservationClass] = frozenset(
            map(lambda x: parse_configuration(ObservationClass, x), obs_class)
        )
        self.program_types: FrozenSet[ProgramTypes] = frozenset(
            map(lambda x: parse_configuration(ProgramTypes, x), prg_type)
        )
        self.time_slot_length: TimeDelta = TimeDelta(time_slot_length * u.min)

    def __iter__(self):
        return iter((self.time_slot_length,
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
        if algorithm_name.upper() == 'DUMMY':
            return DummyOptimizer()
        elif algorithm_name.upper() == 'GREEDYMAX':
            return GreedyMaxOptimizer()
        else:
            raise ConfigurationError('Optimizer', config.optimizer.name)

    def __iter__(self):
        return iter([self.algorithm])



class Blueprints:
    collector: CollectorBlueprint = CollectorBlueprint(config.collector.observation_classes,
                                                       config.collector.program_types,
                                                       config.collector.time_slot_length)
    optimizer: OptimizerBlueprint = OptimizerBlueprint(config.optimizer.name)

