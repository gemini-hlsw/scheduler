import os

from api.observatory.gemini import GeminiProperties
from api.programprovider.ocs import read_ocs_zipfile, OcsProgramProvider
from common.output import print_collector_info
from components.collector import *
from components.selector import Selector


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
        sites=ALL_SITES,
        semesters={Semester(2018, SemesterHalf.B)},
        program_types={ProgramTypes.Q, ProgramTypes.LP, ProgramTypes.FT, ProgramTypes.DD},
        obs_classes={ObservationClass.SCIENCE, ObservationClass.PROGCAL, ObservationClass.PARTNERCAL}
    )
    collector.load_programs(program_provider=OcsProgramProvider(),
                            data=programs)

    # Output the state of and information calculated by the Collector.
    print_collector_info(collector, samples=60)

    selector = Selector(
        collector=collector,
        properties=GeminiProperties
    )

    # Execute the Selector.
    night_indices = np.array([0, 1])
    results = selector.select()
    
    # Output the data in a spreadsheet.
    # for program in collector._programs.values():
    #    if program.id == 'GS-2018B-Q-101':
    #        atoms_to_sheet(program)
