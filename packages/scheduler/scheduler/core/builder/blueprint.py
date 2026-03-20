# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from abc import ABC
from typing import final, Any, FrozenSet, List, Type
from enum import Enum
from typing import Optional

from astropy.time import TimeDelta
import astropy.units as u

from lucupy.minimodel.observation import ObservationClass
from lucupy.minimodel.program import ProgramTypes

from scheduler.config import config, ConfigurationError
from scheduler.core.components.optimizer.optimizers import BaseOptimizer, Optimizers


__all__ = [
    'parse_configuration',
    'Blueprint',
    'Blueprints',
    'CollectorBlueprint',
    'SelectorBlueprint',
    'OptimizerBlueprint',
]


def parse_configuration(enum_class: Type[Enum], value: str) -> Any:
    """General parser for config.yml

    Args:
        enum_class (Type): Enum corresponding to the setting to parse
        value (str): Value on config.yml

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


class Blueprint(ABC):
    """Base class for a Blueprint.
    """
    pass


@final
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


@final
class SelectorBlueprint(Blueprint):
    """Blueprint for the Selector.
    This is based on the configuration in config.yml used to specify the buffer time to determine by how much programs
    may go over their time limit.
    """
    def __init__(self,
                 buffer_type_str: str,
                 buffer_amount: Optional[float]):
        self.buffer_type_str = buffer_type_str
        self.buffer_amount = buffer_amount

    def __iter__(self):
        return iter((self.buffer_type_str,
                     self.buffer_amount))


@final
class OptimizerBlueprint(Blueprint):
    """Blueprint for the Optimizer.
    This is based on the configuration in config.yml.
    """

    def __init__(self, algorithm: str) -> None:
        self.algorithm = OptimizerBlueprint._parse_optimizer(algorithm)

    @staticmethod
    def _parse_optimizer(algorithm_name: str) -> BaseOptimizer:
        try:
            instantiator = Optimizers[algorithm_name.upper()]
            return instantiator.value()
        except KeyError:
            raise ConfigurationError('Optimizer', config.optimizer.name)

    def __iter__(self):
        return iter((self.algorithm,))


@final
class Blueprints:
    collector: CollectorBlueprint = CollectorBlueprint(config.collector.observation_classes,
                                                       config.collector.program_types,
                                                       config.collector.time_slot_length)
    selector: SelectorBlueprint = SelectorBlueprint(config.selector.buffer_type,
                                                    config.selector.buffer_amount)
    optimizer: OptimizerBlueprint = OptimizerBlueprint(config.optimizer.name)
