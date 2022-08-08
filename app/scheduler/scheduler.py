
import signal
import os

from omegaconf import DictConfig
from astropy.time import Time, TimeDelta
import astropy.units as u

from app.api.programprovider.ocs import read_ocs_zipfile, OcsProgramProvider
from app.api.observatory.abstract import ObservatoryProperties
from app.api.observatory.gemini import GeminiProperties
from app.components.collector import Collector
from app.components.selector import Selector
from app.components.optimizer import Optimizer
from app.components.optimizer.dummy import DummyOptimizer
from common.output import print_plans
from common.minimodel import Site, ALL_SITES, Semester, ProgramTypes, ObservationClass, SemesterHalf
from app.plan_manager import PlanManager
from app.config import config


class Scheduler:
    def __init__(self, start_time: Time, end_time: Time):
        self.config = config
        self.start_time = start_time
        self.end_time = end_time
        self.plan_manager = PlanManager.instance()

    def __call__(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        ObservatoryProperties.set_properties(GeminiProperties)
        
        programs = read_ocs_zipfile(os.path.join(os.getcwd(), 'app', 'data', '2018B_program_samples.zip'))

        # Create the Collector and load the programs.
        print('Loading programs...')
        # Parse configs
        time_slot_length = TimeDelta(self.config.scheduler.time_slot_length * u.min)
        sites = frozenset(list(map(eval, self.config.scheduler.sites)))\
            if isinstance(self.config.scheduler.sites, list) else eval(self.config.scheduler.sites)
        semesters = set(map(eval, self.config.collector.semesters))
        program_types = set(map(eval, self.config.collector.program_types))
        obs_classes = set(map(eval, self.config.collector.observation_classes))

        collector = Collector(
            start_time=self.start_time,
            end_time=self.end_time,
            time_slot_length=time_slot_length,
            sites=sites,
            semesters=frozenset(semesters),
            program_types=frozenset(program_types),
            obs_classes=frozenset(obs_classes)
        )
        collector.load_programs(program_provider=OcsProgramProvider(),
                                data=programs)

        if Site.GS in sites or Site.GN in sites:
            ObservatoryProperties.set_properties(GeminiProperties)
        else:
            raise NotImplementedError('Only Gemini is supported.')
        
        selector = Selector(collector=collector)

        # Execute the Selector.
        # Not sure the best way to display the output.
        selection = selector.select()
        # Execute the Optimizer.
        dummy = DummyOptimizer()
        optimizer = Optimizer(selection, algorithm=dummy)
        plans = optimizer.schedule()

        print_plans(plans)
        self.plan_manager.set_plans(plans)


def build_scheduler():
    # TODO: Default values for now but this should be:
    # start_time = Time(datetime.now(), scale='utc)
    # end_time = Time(datetime.now() + rest_of_the_night, scale='utc')
    start = Time("2018-10-01 08:00:00", format='iso', scale='utc')
    end = Time("2018-10-03 08:00:00", format='iso', scale='utc')
    return Scheduler(start, end)
