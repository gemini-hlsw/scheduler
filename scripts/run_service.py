from app import App
from mock.observe import Observe
import logging
from astropy.time import Time, TimeDelta
import astropy.units as u
from app.config import SchedulerConfig, CollectorConfig, SelectorConfig
from api.observatory.gemini import GeminiProperties
from common.minimodel import ALL_SITES, Semester, SemesterHalf, ProgramTypes, ObservationClass
import asyncio

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

    app = App(config)
    asyncio.run(app.run())
    # Observe.start()
    
