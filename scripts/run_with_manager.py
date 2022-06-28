
import asyncio
import logging
from astropy.time import Time, TimeDelta
import astropy.units as u
from app.process_manager import TaskType, ProcessManager, Scheduler
from app.process_manager import SchedulerConfig, CollectorConfig, SelectorConfig
from api.observatory.gemini import GeminiProperties
from common.minimodel import ALL_SITES, Semester, SemesterHalf, ProgramTypes, ObservationClass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Create Scheduler
    collector_config = CollectorConfig({Semester(2018, SemesterHalf.B)},
                                    {ProgramTypes.Q, ProgramTypes.LP, ProgramTypes.FT, ProgramTypes.DD},
                                    {ObservationClass.SCIENCE, ObservationClass.PROGCAL, ObservationClass.PARTNERCAL}
                                    )

    selector_config = SelectorConfig(GeminiProperties)

    config = SchedulerConfig(Time("2018-10-01 08:00:00", format='iso', scale='utc'),
                             Time("2018-10-03 08:00:00", format='iso', scale='utc'),
                             TimeDelta(1.0 * u.min),
                             ALL_SITES,
                             collector_config,
                             selector_config)

    scheduler = Scheduler(config)

    # Manager params
    mode = TaskType.STANDARD  # Type of runner is going to be working
    size = 2  # number of processes
    timeout = 60 * 60  # max time to wait for a process to finish
    period = 2

    manager = ProcessManager(size, timeout)
    try:
        asyncio.run(manager.run(scheduler, period, mode))
    except RuntimeError:
        # Likely you pressed Ctrl-C...
        ...
