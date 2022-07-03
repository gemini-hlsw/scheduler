import asyncio
import signal
from datetime import datetime
from .process_manager import ProcessManager, TaskType
from .scheduler import Scheduler

TASK_QUEUE = []


class App:
    def __init__(self, config):
        self.config = config
        self.manager = ProcessManager()

    def build_scheduler(self):
        # TODO: This needs to be modified to use more robust way to build schedulers
        # Right are all the same but with different configs (or parameters)
        return Scheduler(self.config)
    
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
            scheduler = self.build_scheduler()
            self.manager.add_task(datetime.now(), scheduler, mode)

            await asyncio.sleep(10)
