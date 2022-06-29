
import signal
import os

from omegaconf import DictConfig
from astropy.time import Time, TimeDelta
import astropy.units as u

from api.programprovider.ocs import read_ocs_zipfile, OcsProgramProvider
from api.observatory.gemini import GeminiProperties
from components.collector import Collector
from components.selector import Selector
from components.optimizer import Optimizer
from components.optimizer.dummy import DummyOptimizer
from common.output import print_plans
from common.minimodel import (Site,
                              ALL_SITES,
                              Semester,
                              SemesterHalf,
                              ProgramTypes,
                              ObservationClass)


class Scheduler():
    def __init__(self, config: DictConfig, start_time: Time, end_time: Time):
        self.config = config
        self.start_time = start_time
        self.end_time = end_time

    def __call__(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        programs = read_ocs_zipfile(os.path.join('..', 'data', '2018B_program_samples.zip'))

        # Create the Collector and load the programs.

        # Parse config
        time_slot_length = TimeDelta(self.config.scheduler.time_slot_length * u.min)
        sites = frozenset(list(map(eval, self.config.scheduler.sites))) if isinstance(self.config.scheduler.sites, list) else eval(self.config.scheduler.sites)
        semesters = set(map(eval, self.config.collector.semesters))
        program_types = set(map(eval, self.config.collector.program_types))
        obs_classes = set(map(eval, self.config.collector.observation_classes))

        collector = Collector(
            start_time=self.start_time,
            end_time=self.end_time,
            time_slot_length=time_slot_length,
            sites=sites,
            semesters=semesters,
            program_types=program_types,
            obs_classes=obs_classes
        )
        collector.load_programs(program_provider=OcsProgramProvider(),
                                data=programs)

        if Site.GS in sites or Site.GN in sites:
            properties = GeminiProperties
        else:
            raise NotImplementedError('Only Gemini is supported')
        
        selector = Selector(collector=collector,
                            properties=properties)

        # Execute the Selector.
        # Not sure the best way to display the output.
        selection = selector.select()
        # Execute the Optimizer.
        dummy = DummyOptimizer()
        optimizer = Optimizer(selection, algorithm=dummy)
        plans = optimizer.schedule()
        print_plans(plans)
