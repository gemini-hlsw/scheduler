import asyncio
import signal
import backoff
from datetime import datetime
from gql.tranport.websockets import WebsocketsTransport
from .process_manager import ProcessManager, TaskType
from .scheduler import Scheduler
from astropy.time import Time


class App:
    def __init__(self, config):
        self.config = config
        self.manager = ProcessManager(size=config.process_manager.size,
                                      timeout=config.process_manager.timeout)
        self.transport = WebsocketsTransport(url="wss://YOUR_URL")

    def build_scheduler(self, start_time: Time, end_time: Time):
        # TODO: This needs to be modified to use more robust way to build schedulers
        # Right are all the same but with different configs (or parameters)
        return Scheduler(self.config, start_time, end_time)
    
    @backoff.on_exception(backoff.expo, Exception, max_time=300)
    async def run(self):
        done = asyncio.Event()

        def shutdown():
            done.set()
            self.manager.shutdown()
            asyncio.get_event_loop().stop()
        
        asyncio.get_event_loop().add_signal_handler(signal.SIGINT, shutdown)

        while not done.is_set():

            # Observe interaction
            mode = TaskType.STANDARD
            start = Time("2018-10-01 08:00:00", format='iso', scale='utc')
            end = Time("2018-10-03 08:00:00", format='iso', scale='utc')
            scheduler = self.build_scheduler(start, end)
            self.manager.add_task(datetime.now(), scheduler, mode)

            await asyncio.sleep(10)
