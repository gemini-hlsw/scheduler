# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import os
import signal
from abc import ABC, abstractmethod
from enum import Enum

from astropy.time import Time

from typing import ClassVar, FrozenSet, Iterable, Callable, Optional, NoReturn
from lucupy.minimodel.observation import ObservationStatus, Observation
from lucupy.minimodel.program import Program

from app.config import config_collector
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

    def build_collector(self, start: Time, end: Time):
        # Base collector
        return Collector(
            start_time=start,
            end_time=end,
            time_slot_length=config_collector.time_slot_length,
            sites=config_collector.sites,
            semesters=config_collector.semesters,
            program_types=config_collector.program_types,
            obs_classes=config_collector.obs_classes
        )
    def __str__(self) -> str:
        return self.__class__.__name__


class SimulationMode(SchedulerMode):
    def schedule(self, start: Time, end: Time):
        ...

    def schedule(self, start: Time, end: Time):
       pass

class ValidationMode(SchedulerMode):
    """
    
    """
    
    _obs_statuses_to_ready: ClassVar[FrozenSet[ObservationStatus]] = (
        frozenset([ObservationStatus.ONGOING, ObservationStatus.OBSERVED])
    )
    """The default observations to set to READY in Validation mode.
    """


    @staticmethod
    def clear_observation_info(programs: Iterable[Program],
                               obs_statuses_to_ready: FrozenSet[ObservationStatus] = _obs_statuses_to_ready,
                               program_filter: Optional[Callable[[Program], bool]] = None,
                               observation_filter: Optional[Callable[[Observation], bool]] = None) -> NoReturn:
        """
        Iterate over the loaded programs and clear the information associated with the observations.
        This is done when the Scheduler is run in Validation mode in order to start with a set of fresh Observations.

        This consists of:
        1. Setting observation statuses that are in obs_statuses_to_ready to READY (default: ONGOING or OBSERVED).
        2. Setting used times to 0 for observations.

        The Observations affected by this are those with statuses specified by the parameter observations_to_process.
        Additional filtering may be done by specifying optional filters for programs and a filter for observations.
        """
        program_candidates = (programs if program_filter is None
                              else (p for p in programs if program_filter(p)))

        for program in program_candidates:
            observation_candidates = (program.observations() if observation_filter is None
                                      else (o for o in program.observations() if observation_filter(o)))
            for observation in observation_candidates:
                # Clear the time used across the sequence regardless of status.
                for atom in observation.sequence:
                    atom.prog_time = timedelta()
                    atom.part_time = timedelta()

                # Change the status of observations with indicated status to READY.
                if observation.status in obs_statuses_to_ready:
                    observation.status = ObservationStatus.READY

    def build_collector(self, programs):
        base_collector =  super().build_collector()
        base_collector.load_programs(program_provider=OcsProgramProvider(),
                                     data=programs)
        ValidationMode.clear_observation_info(programs)    

    def schedule(self, start: Time, end: Time):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        programs = read_ocs_zipfile(os.path.join(ROOT_DIR, 'app', 'data', '2018B_program_samples.zip'))

        # Create the Collector and load the programs.
        print('Loading programs...')
        collector = Collector(
            start_time=start,
            end_time=end,
            time_slot_length=config_collector.time_slot_length,
            sites=config_collector.sites,
            semesters=config_collector.semesters,
            program_types=config_collector.program_types,
            obs_classes=config_collector.obs_classes
        )
        collector.load_programs(program_provider=OcsProgramProvider(),
                                data=programs)

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
    def schedule(self, start: Time, end: Time):
        ...

    def schedule (self, start: Time, end: Time):
       pass

class SchedulerModes(Enum):
    OPERATION = OperationMode()
    SIMULATION = SimulationMode()
    VALIDATION = ValidationMode()
