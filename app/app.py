import asyncio
import signal
from datetime import datetime

from .process_manager import ProcessManager, TaskType
from .scheduler import Scheduler
from common.queries import Session, observation_update, program_update, target_update
from astropy.time import Time


class App:
    def __init__(self, config):
        self.config = config
        self.manager = ProcessManager(size=config.process_manager.size,
                                      timeout=config.process_manager.timeout)
        self.session = Session(url=config.graphql.url)

    def build_scheduler(self, start_time: Time, end_time: Time):
        # TODO: This needs to be modified to use more robust way to build schedulers
        # Right are all the same but with different configs (or parameters)
        return Scheduler(self.config, start_time, end_time)

    async def run(self):
        done = asyncio.Event()

        def shutdown():
            done.set()
            [t.cancel() for t in asyncio.all_tasks()]
            self.manager.shutdown()
            asyncio.get_event_loop().stop()
        
        asyncio.get_event_loop().add_signal_handler(signal.SIGINT, shutdown)

        while not done.is_set():
            
            resp = await self.session.subscribe_all()
            if resp:
                print(resp)
                # Run new Schedule
                mode = TaskType.STANDARD
                start = Time("2018-10-01 08:00:00", format='iso', scale='utc')
                end = Time("2018-10-03 08:00:00", format='iso', scale='utc')
                scheduler = self.build_scheduler(start, end)
                self.manager.add_task(datetime.now(), scheduler, mode)
            
            await asyncio.sleep(10)
