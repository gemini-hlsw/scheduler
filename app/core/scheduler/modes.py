# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import functools
from abc import ABC
from datetime import timedelta
from enum import Enum
from typing import ClassVar, FrozenSet, Iterable, Callable, Optional, NoReturn
from app.config import ConfigurationError, config

from lucupy.minimodel.observation import ObservationStatus, Observation


class SchedulerMode(ABC):
    """Base Scheduler Mode.

    Right now the magic method `__str__` is the only thing that inheritance from this.
    Also good for duck typing.
    """
    def __str__(self) -> str:
        return self.__class__.__name__


class SimulationMode(SchedulerMode):
    pass


class ValidationMode(SchedulerMode):
    """Validation mode is used for validate the proper functioning

    Attributes:
        _obs_statuses_to_ready (ClassVar[FrozenSet[ObservationStatus]]): 
            A set of statuses that show the observation is Ready.
    """

    # The default observations to set to READY in Validation mode.
    _obs_statuses_to_ready: ClassVar[FrozenSet[ObservationStatus]] = (
        frozenset([ObservationStatus.ONGOING, ObservationStatus.OBSERVED])
    )

    @staticmethod
    def _clear_observation_info(obs: Iterable[Observation],
                                obs_statuses_to_ready: FrozenSet[ObservationStatus],
                                observation_filter: Optional[Callable[[Observation], bool]] = None) -> NoReturn:
        """
        Given a single observation, clear the information associated with the observation.
        This is done when the Scheduler is run in Validation mode in order to start with a fresh observation.

        This consists of:
        1. Setting an observation status that is in obs_statuses_to_ready to READY (default: ONGOING or OBSERVED).
        2. Setting used times to 0 for the observation.

        Additional filtering may be done by specifying an optional filter for observations.
        """
        if observation_filter is not None:
            filtered_obs = (o for o in obs if observation_filter(o))
        else:
            filtered_obs = obs

        for o in filtered_obs:
            for atom in o.sequence:
                atom.prog_time = timedelta()
                atom.part_time = timedelta()

            if o.status in obs_statuses_to_ready:
                o.status = ObservationStatus.READY

    @staticmethod
    def build_collector_with_clear_info(func: Callable):
        """Decorator that modifies the build_collector method in SchedulerBuilder.

        Args:
            func (Callable): SchedulerBuilder.build_collector original method
        Returns:
            collector (Collector): Returns modified collector with clear info in the observations.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            collector = func(*args, **kwargs)
            ValidationMode._clear_observation_info(collector.get_all_observations(),
                                                   ValidationMode._obs_statuses_to_ready)
            return collector
        return wrapper


class OperationMode(SchedulerMode):
    pass


class SchedulerModes(Enum):
    """Scheduler modes available:

    - OPERATION
    - SIMULATION
    - VALIDATION

    """
    OPERATION = OperationMode()
    SIMULATION = SimulationMode()
    VALIDATION = ValidationMode()


def dispatch_with(mode: str):
    """Decorator that allows different behavior on the whole or parts of
    the Scheduler by modifying the building methods in the SchedulerBuilder.

    Args:
        mode (str): Mode string from config.yml.

    Raises:
        ValueError: If config name is not found in SchedulerModes.

    Returns:
        SchedulerBuild: Modify version of the SchedulerBuild depending on the mode. 
    """
    # Setup scheduler mode
    try:
        mode = SchedulerModes[config.mode]
    except ValueError:
        raise ConfigurationError(f'Mode "{config.mode}" is invalid.')

    def decorator_dispatcher(cls):
        @functools.wraps(cls)
        def scheduler_wrapper(*args, **kwargs):
            if mode is SchedulerModes.VALIDATION:
                cls.build_collector = mode.value.build_collector_with_clear_info(cls.build_collector)
            return scheduler_wrapper
        return scheduler_wrapper
    return decorator_dispatcher
