from abc import ABC, abstractmethod
from astropy.time import Time
import os
import signal

from app.config import config_collector
from app.core.components.collector import Collector
from app.core.components.optimizer import Optimizer
from app.core.components.optimizer.dummy import DummyOptimizer
from app.core.components.selector import Selector
from app.core.programprovider.ocs import read_ocs_zipfile, OcsProgramProvider
from definitions import ROOT_DIR
from app.db.planmanager import PlanManager


class SchedulerMode(ABC):

    @abstractmethod
    def schedule(self, start: Time, end: Time):
        pass
    
    def __str__(self) -> str:
        return self.__class__.__name__

class SimulationMode(SchedulerMode):

    def schedule(self):
       pass

class ValidationMode(SchedulerMode):

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
        #print(f'{PlanManager._plans=}')

class OperationMode(SchedulerMode):

    def schedule(self):
       ...
