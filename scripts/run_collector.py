import os

from api.ocs import read_ocs_zipfile, OcsProgramProvider
from common.output import print_collector_info
from collector import *


if __name__ == '__main__':
    # Reduce logging to ERROR only to display the tqdm bars nicely.
    logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(level=logging.ERROR)

    # Read in a list of JSON data
    programs = read_ocs_zipfile(os.path.join('..', 'data', '2018B_program_samples.zip'))

    # Create the Collector and load the programs.
    collector = Collector(
        start_time=Time("2018-10-01 08:00:00", format='iso', scale='utc'),
        end_time=Time("2018-10-03 08:00:00", format='iso', scale='utc'),
        time_slot_length=1.0 * u.min,
        sites={Site.GN, Site.GS},
        semesters={Semester(2018, SemesterHalf.B)},
        program_types={ProgramTypes.Q, ProgramTypes.LP, ProgramTypes.FT, ProgramTypes.DD},
        obs_classes={ObservationClass.SCIENCE, ObservationClass.PROGCAL, ObservationClass.PARTNERCAL}
    )
    collector.load_programs(program_provider=OcsProgramProvider(),
                            data=programs)

    # Output the state of and information calculated by the Collector.
    print_collector_info(collector, samples=60)
