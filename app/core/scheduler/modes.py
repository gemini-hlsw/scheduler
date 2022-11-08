# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import os
import signal
import functools
from abc import ABC, abstractmethod
from datetime import timedelta
from enum import Enum
from typing import ClassVar, FrozenSet, Iterable, Callable, Optional, NoReturn

from astropy.time import Time

from lucupy.minimodel.observation import ObservationStatus, Observation
from lucupy.minimodel.program import Program

from app.core.components.builder import Blueprints
from app.core.components.collector import Collector
from app.core.components.optimizer import Optimizer
from app.core.components.optimizer.dummy import DummyOptimizer
from app.core.components.selector import Selector
from app.core.programprovider.ocs import read_ocs_zipfile, OcsProgramProvider
from app.db.planmanager import PlanManager
from definitions import ROOT_DIR


class SchedulerMode(ABC):

    @abstractmethod
    def schedule(self, start: Time, end: Time):
        pass
    
    def __str__(self) -> str:
        return self.__class__.__name__


class SimulationMode(SchedulerMode):
    def schedule(self, start: Time, end: Time):
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
        if observation_filter is not None and not observation_filter(obs):
            return

        for o in obs:
            for atom in o.sequence:
                atom.prog_time = timedelta()
                atom.part_time = timedelta()

            if o.status in obs_statuses_to_ready:
                o.status = ObservationStatus.READY

    def schedule(self, start: Time, end: Time):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        programs = read_ocs_zipfile(os.path.join(ROOT_DIR, 'app', 'data', '2018B_program_samples.zip'))

        # Create the Collector and load the programs.
        print('Loading programs...')
        collector = Collector(*Blueprints.collector)
        collector.load_programs(program_provider=OcsProgramProvider(),
                                data=programs)
        ValidationMode._clear_observation_info(collector.get_all_observations(), ValidationMode._obs_statuses_to_ready)
        selector = Selector(collector=collector)

        # Execute the Selector.
        # Not sure the best way to display the output.
        selection = selector.select()
        # Execute the Optimizer.
        dummy = DummyOptimizer()
        optimizer = Optimizer(selection, algorithm=dummy)
        plans = optimizer.schedule()

        # Save to database
        PlanManager.set_plans(plans)
    


class OperationMode(SchedulerMode):

    def schedule (self, start: Time, end: Time):
       pass

class SchedulerModes(Enum):
    OPERATION = OperationMode()
    SIMULATION = SimulationMode()
    VALIDATION = ValidationMode()
