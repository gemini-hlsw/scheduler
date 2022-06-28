
import signal
import os
from api.programprovider.ocs import read_ocs_zipfile, OcsProgramProvider
from components.collector import Collector
from components.selector import Selector
from components.optimizer import Optimizer
from components.optimizer.dummy import DummyOptimizer
from common.output import print_plans
from .config import SchedulerConfig



class Scheduler():
    def __init__(self, config: SchedulerConfig):
        self.config = config

    def __call__(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        programs = read_ocs_zipfile(os.path.join('..', 'data', '2018B_program_samples.zip'))

        # Create the Collector and load the programs.
        collector = Collector(
            start_time=self.config.start_time,
            end_time=self.config.end_time,
            time_slot_length=self.config.time_slot_length,
            sites=self.config.sites,
            semesters=self.config.collector.semesters,
            program_types=self.config.collector.program_types,
            obs_classes=self.config.collector.obs_classes
        )
        collector.load_programs(program_provider=OcsProgramProvider(),
                                data=programs)

        selector = Selector(collector=collector,
                            properties=self.config.selector.properties)

        # Execute the Selector.
        # Not sure the best way to display the output.
        selection = selector.select()
        # Execute the Optimizer.
        dummy = DummyOptimizer()
        optimizer = Optimizer(selection, algorithm=dummy)
        plans = optimizer.schedule()
        print_plans(plans)
