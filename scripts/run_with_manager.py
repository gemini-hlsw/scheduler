
import asyncio
import signal
from process_manager.task import TaskType
from process_manager.manager import ProcessManager
from process_manager.scheduler import SchedulerConfig, CollectorConfig
from api.observatory.gemini import GeminiProperties

from common.minimodel import ALL_SITES, Semester, SemesterHalf, ProgramTypes, ObservationClass


async def main(args):
    done = asyncio.Event()
    manager = ProcessManager(args.size, args.timeout)
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

    def shutdown():
        done.set()
        manager.shutdown()
        asyncio.get_event_loop().stop()

    asyncio.get_event_loop().add_signal_handler(signal.SIGINT, shutdown)

    while not done.is_set():
        
        logging.info(f"Scheduling a job for {task}")
        manager.add_task(config, mode)
        if args.period == 0:
            # random case #
            await asyncio.sleep(randint(1, 10))
        else:
            await asyncio.sleep(args.period)



if __name__ == '__main__':

    try:
        asyncio.run(main())
    finally:
       