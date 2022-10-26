from abc import ABC, abstractmethod
import astropy.units as u
from astropy.time import Time, TimeDelta
import os
import signal

# These depdencies looks like are not being reference but are necesary in runtime when eval() is applied
from lucupy.minimodel import Site, ALL_SITES, Semester, ProgramTypes, ObservationClass, SemesterHalf
from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties

from app.config import config
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
        # Parse configs
        time_slot_length = TimeDelta(config.scheduler.time_slot_length * u.min)
        sites = frozenset(list(map(eval, config.scheduler.sites))) \
            if isinstance(config.scheduler.sites, list) else eval(config.scheduler.sites)
        semesters = set(map(eval, config.collector.semesters))
        program_types = set(map(eval, config.collector.program_types))
        obs_classes = set(map(eval, config.collector.observation_classes))

        collector = Collector(
            start_time=start,
            end_time=end,
            time_slot_length=time_slot_length,
            sites=sites,
            semesters=frozenset(semesters),
            program_types=frozenset(program_types),
            obs_classes=frozenset(obs_classes)
        )
        collector.load_programs(program_provider=OcsProgramProvider(),
                                data=programs)

        '''
        if Site.GS in sites or Site.GN in sites:
            ObservatoryProperties.set_properties(GeminiProperties)
        else:
            raise NotImplementedError('Only Gemini is supported.')
        '''
        

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
