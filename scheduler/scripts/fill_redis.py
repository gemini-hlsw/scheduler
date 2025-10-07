# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from inspect import isclass
from pathlib import Path
import asyncio

import numpy as np
from astropy.time import Time
from lucupy.minimodel import NightIndex
from lucupy.minimodel.semester import Semester
from lucupy.minimodel.site import ALL_SITES, Site
from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties
from lucupy.types import ZeroTime

from definitions import ROOT_DIR
from scheduler.core.builder.blueprint import CollectorBlueprint
from scheduler.core.programprovider.abstract import ProgramProvider
from scheduler.core.programprovider.ocs import ocs_program_data, OcsProgramProvider
from scheduler.core.components.collector import Collector

from scheduler.core.sources.sources import Sources
from scheduler.services.redis_client import redis_client
from scheduler.services.visibility.calculator import VisibilityCalculator


async def calculate_and_save():
    start = Time("2018-08-01 08:00:00", format='iso', scale='utc')
    end = Time("2019-01-31 08:00:00", format='iso', scale='utc')

    ObservatoryProperties.set_properties(GeminiProperties)

    # Create the Collector and load the programs.
    collector_blueprint = CollectorBlueprint(
        ['SCIENCE', 'PROGCAL', 'PARTNERCAL'],
        ['Q', 'LP', 'FT', 'DD', 'C'],
        1.0
    )

    semesters = frozenset([Semester.find_semester_from_date(start.datetime),
                           Semester.find_semester_from_date(end.datetime)])

    sem, = semesters

    night_indices = frozenset(NightIndex(idx) for idx in range(1))
    sites = ALL_SITES
    programs_ids = Path(ROOT_DIR) / 'scheduler' / 'data' / 'program_ids.redis.txt'

    # check if path exist and read
    f_programs = None
    if programs_ids:
        programs_path = Path(programs_ids)

        if programs_path.exists():
            f_programs = programs_path
        else:
            raise ValueError(f'Path {programs_path} does not exist.')

    sources = Sources()
    # Create the Collector, load the programs, and zero out the time used by the observations.
    collector = Collector(
        start_vis_time=start,
        end_vis_time=end,
        night_start_time=None,
        night_end_time=None,
        num_of_nights=1,
        sites=sites,
        semesters=semesters,
        sources=sources,
        time_slot_length=collector_blueprint.time_slot_length,
        program_types=collector_blueprint.program_types,
        obs_classes=collector_blueprint.obs_classes
    )

    program_provider_class = OcsProgramProvider
    data = ocs_program_data(f_programs)

    if not (isclass(program_provider_class) and issubclass(program_provider_class, ProgramProvider)):
        raise ValueError('Collector load_programs requires a ProgramProvider class as the second argument')
    program_provider = program_provider_class(collector.obs_classes, sources)

    # Purge the old programs and observations.
    Collector._programs = {}

    # Keep a list of the observations for parallel processing.
    parsed_observations = []

    # Read in the programs.
    # Count the number of parse failures.
    bad_program_count = 0

    for json_program in data:
        try:
            if len(json_program.keys()) != 1:
                msg = f'JSON programs should only have one top-level key: {" ".join(json_program.keys())}'
                raise ValueError(msg)

            # Extract the data from the JSON program. We do not need the top label.
            data = next(iter(json_program.values()))
            program = program_provider.parse_program(data)

            # If program could not be parsed, skip. This happens in one of three cases:
            # 1. Program semester cannot be determined from ID.
            # 2. Program type cannot be determined from ID.
            # 3. Program root group is empty.
            if program is None:
                continue

            # TODO: improve this. Pass the semesters into the program_provider and return None as soon
            # TODO: as we know that the program is not from a semester in which we are interested.
            # If program semester is not in the list of specified semesters, skip.
            if program.semester is None or program.semester not in semesters:
                continue

            # If a program has no time awarded, then we will get a divide by zero in scoring, so skip it.
            if program.program_awarded() == ZeroTime:
                continue

            collector._programs[program.id] = program

            # Set the observation IDs for this program.
            # We only want the observations that are located at the sites supported by the collector.
            # TODO: In GPP, if an AndGroup exists where the observations are not all from the same site, then
            # TODO: this should be an error.
            # TODO: In the case of an OrGroup, we only want:
            # TODO: 1. The branches that are OrGroups and are nonempty (i.e. have obs).
            # TODO: 2. The branches that are AndGroups and are nonempty (i.e. all obs are from the same site).
            # TODO: Applying this logic recursively should ensure only Groups that can be completed are included.
            site_supported_obs = [obs for obs in program.observations() if obs.site in sites]
            if site_supported_obs:
                collector._observations_per_program[program.id] = frozenset(obs.id for obs in site_supported_obs)
                parsed_observations.extend((program.id, obs) for obs in site_supported_obs)

        except Exception as e:
            bad_program_count += 1

    # vis_calc = VisibilityCalculator()
    vis_table = {}
    for program_id, obs in parsed_observations:
        # Check for a base target in the observation: if there is none, we cannot process.
        # For ToOs, this may be the case.
        base = obs.base_target()
        if base is None:
            continue

        program = collector.get_program(program_id)
        if program is None:
            raise RuntimeError(f'Could not find program {program_id.id} for observation {obs.id.id}.')

        # Record the observation and target for this obs id.
        collector._observations[obs.id] = obs, base

        tw = Collector._process_timing_windows(program, obs)

        if obs.site not in collector.night_events:
            raise ValueError(f'Requested obs {obs.id.id} target info for site {obs.site}, which is not included.')
        night_events = collector.night_events[obs.site]

        # Get the night configurations (for resources)
        nc = collector.night_configurations(obs.site, np.arange(collector.num_nights_calculated))

        print('Calculating visibility for obs: ', obs.id)
        program = collector.get_program(obs.id.program_id())

        vis_table[obs.id.id] = VisibilityCalculator.calculate_visibility(obs,
                                                                         base,
                                                                         program,
                                                                         night_events,
                                                                         nc,
                                                                         collector.time_grid,
                                                                         tw,
                                                                         collector.time_slot_length)

    main_key = f"{sem}-{collector.time_slot_length.to_value('min')}min"
    print('Setting key: ', main_key)
    await redis_client.set_whole_dict(main_key, vis_table)

    # main_key = f"2018b1.0min"
    import time

    s = time.perf_counter()
    vis_table = await redis_client.get_whole_dict(main_key)
    e = time.perf_counter()
    print('Time to retrieve all visibilities: ', e - s, 's')
    assert len(vis_table['GN-2018B-LP-15-4'].keys()) == 184  # match number of days in the semester
    assert len(vis_table.keys()) == len(parsed_observations)  # match number of observations parsed

if __name__ == '__main__':
    """
    The script has to be run by semester modifying the start and end variables to 
    the correct dates for each semester half. Default is 2018B.
    `program_ids` can be supply (as a file path) so the amount of programs stored can change. 
    """
    asyncio.run(calculate_and_save())
