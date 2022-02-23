import os
import json
from typing import NoReturn

from astropy import units as u

from api.abstract import ProgramProvider
from api.ocs import OcsProgramProvider
from common.minimodel import Group, Observation, Program
from collector import Collector, NightEventsManager


def print_program_from_provider(filename=os.path.join('data', 'GN-2018B-Q-101.json'),
                                provider: ProgramProvider = OcsProgramProvider) -> NoReturn:
    """
    Using a specified JSON file and a ProgramProvider, read in the program
    and print it.

    TODO: Could pass in JSON data instead, as GppProgramProvider will not produce files.
    """
    with open(filename, 'r') as f:
        data = json.loads(f.read())

    program = provider.parse_program(data['PROGRAM_BASIC'])
    print_program(program)


def print_program(program: Program) -> NoReturn:
    """
    Print the high-level information about a program in human semi-readable format
    to give an idea as to its structure.
    """
    print(f'Program: {program.id}')

    def sep(depth: int) -> str:
        return '----- ' * depth

    def print_observation(depth: int, obs: Observation) -> NoReturn:
        print(f'{sep(depth)} Observation: {obs.id}')
        for atom in obs.sequence:
            print(f'{sep(depth + 1)} {atom}')

    def print_group(depth: int, group: Group) -> NoReturn:
        # Is this a subgroup or an observation?
        if isinstance(group.children, Observation):
            print_observation(depth, group.children)
        elif isinstance(group.children, list):
            print(f'{sep(depth)} Group: {group.id}')
            for child in group.children:
                print_group(depth + 1, child)

    # Print the group and atom information.
    print_group(1, program.root_group)


def print_collector_info(collector: Collector, samples: int = 60) -> NoReturn:
    # Output some information.
    print(f'Pre-Collector / Collector running from:')
    print(f'   start time:       {collector.start_time}')
    print(f'   end time:         {collector.end_time}')
    print(f'   time slot length: {collector.time_slot_length}')

    # Print out sampled calculation for every hour as there are far too many results to print in full.
    samples = 60
    time_grid = collector.time_grid
    for site in collector.sites:
        print(f'\n\n+++++ NIGHT EVENTS FOR {site.name} +++++')
        night_events = NightEventsManager.get_night_events(collector.time_grid, collector.time_slot_length, site)
        for idx, jday in enumerate(time_grid):
            start = night_events.local_times[idx][0]
            end = night_events.local_times[idx][-1]
            num_slots = len(night_events.times[idx])
            print(f'* DAY {idx}: {start} to {end}, {num_slots} time slots.')
            print(f'\tmidnight:         {night_events.midnight[idx]}')
            print(f'\tsunset:           {night_events.sunset[idx]}')
            print(f'\tsunrise:          {night_events.sunrise[idx]}')
            print(f'\t12° eve twilight: {night_events.twilight_evening_12[idx]}')
            print(f'\t12° mor twilight: {night_events.twilight_morning_12[idx]}')
            print(f'\tmoonrise:         {night_events.moonrise[idx]}')
            print(f'\tmoonset:          {night_events.moonset[idx]}')
            print(f'\n\tSun information (deg, sampled every {samples} time slots):')
            print(f'\tAlt:    {[a.to_value(u.deg) for a in night_events.sun_alt[idx][::samples]]}')
            print(f'\tAz:     {[a.to_value(u.deg) for a in night_events.sun_az[idx][::samples]]}')
            print(f'\tParAng: {[a.to_value(u.deg) for a in night_events.sun_par_ang[idx][::samples]]}')
            print(f'\n\tMoon information (km or deg, sampled every {samples} time slots):')
            print(f'\tDist:   {[a.to_value(u.km) for a in night_events.moon_dist[idx][::samples]]}')
            print(f'\tAlt:    {[a.to_value(u.deg) for a in night_events.moon_alt[idx][::samples]]}')
            print(f'\tAz:     {[a.to_value(u.deg) for a in night_events.moon_az[idx][::samples]]}')
            print(f'\tParAng: {[a.to_value(u.deg) for a in night_events.moon_par_ang[idx][::samples]]}')
            print(f'\n\tSun-Moon angle (deg, sampled every {samples} time slots):')
            print(f'\t{[a.to_value(u.deg) for a in night_events.sun_moon_ang[idx][::samples]]}')
            print('\n\n')

    # TODO: Using a private property. This will be relocated.
    targets = {(target, obs) for target, obs, _ in Collector._target_info}
    for target, obs in targets:
        print(f'Target {target} in observation {obs}')
    # for k, v in Collector._target_info.items():
    #     target, obs, idx = k
    #     print(f'Observation {obs}, target {target}, night={idx}')
