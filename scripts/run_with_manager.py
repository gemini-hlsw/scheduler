
import asyncio
import signal
import logging
from datetime import datetime
from random import randint
from astropy.time import Time
import astropy.units as u
from process_manager.task import TaskType
from process_manager.manager import ProcessManager
from process_manager.scheduler import Scheduler, SchedulerConfig, CollectorConfig, SelectorConfig
from api.observatory.gemini import GeminiProperties
from common.minimodel import ALL_SITES, Semester, SemesterHalf, ProgramTypes, ObservationClass


async def main(config: SchedulerConfig, size: int, timeout: int, period: int):
    done = asyncio.Event()
    manager = ProcessManager(size, timeout)

    def shutdown():
        done.set()
        manager.shutdown()
        asyncio.get_event_loop().stop()

    asyncio.get_event_loop().add_signal_handler(signal.SIGINT, shutdown)

    while not done.is_set():
        
        # logging.info(f"Scheduling a job for {task}")
        scheduler = Scheduler(config)
        manager.add_task(datetime.now(), scheduler, mode, timeout)
        if period == 0:
            # random case #
            await asyncio.sleep(randint(1, 10))
        else:
            await asyncio.sleep(period)


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)
    mode = TaskType.STANDARD
    collector_config = CollectorConfig({Semester(2018, SemesterHalf.B)},
                                       {ProgramTypes.Q, ProgramTypes.LP, ProgramTypes.FT, ProgramTypes.DD},
                                       {ObservationClass.SCIENCE, ObservationClass.PROGCAL, ObservationClass.PARTNERCAL}
                                       )
    selector_config = SelectorConfig(GeminiProperties)
    config = SchedulerConfig(Time("2018-10-01 08:00:00", format='iso', scale='utc'),
                             Time("2018-10-03 08:00:00", format='iso', scale='utc'),
                             1.0 * u.min,
                             ALL_SITES,
                             collector_config,
                             selector_config)

    # Manager params
    size = 1  # number of processes
    timeout = 60 * 60  # max time to wait for a process to finish
    period = 2000000
    try:
        asyncio.run(main(config, size, timeout, period))
    except RuntimeError:
        # Likely you pressed Ctrl-C...
        ...
