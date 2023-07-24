# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import json
import os
import gzip
from typing import List

from astropy import units as u
from lucupy.minimodel import Atom, Group, Observation, ObservationClass, Program
from openpyxl import Workbook

from scheduler.core.components.collector import Collector, NightEventsManager
from scheduler.core.plans import Plans
from scheduler.core.programprovider.abstract import ProgramProvider
from scheduler.core.programprovider.ocs import OcsProgramProvider


def print_program_from_provider(filename=os.path.join('data', 'GN-2018B-Q-101.json.gz'),
                                provider: ProgramProvider = OcsProgramProvider) -> None:
    """
    Using a specified JSON file and a ProgramProvider, read in the program
    and print it.
    """
    with gzip.open(filename, 'r') as f:
        data = json.loads(f.read())

    program = provider.parse_program(data['PROGRAM_BASIC'])
    program.show()


def print_collector_info(collector: Collector, samples: int = 60) -> None:
    # Output some information.
    print(f'Pre-Collector / Collector running from:')
    print(f'   start time:       {collector.start_time}')
    print(f'   end time:         {collector.end_time}')
    print(f'   time slot length: {collector.time_slot_length.to(u.min)}')

    # Print out sampled calculation for every hour as there are far too many results to print in full.
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

    target_info = sorted((obs_id, Collector.get_base_target(obs_id).name)
                         for obs_id in Collector.get_observation_ids())
    for obs_id, target_name in target_info:
        print(f'Observation {obs_id}, Target {target_name}')


def print_atoms_for_observation(observation: Observation) -> None:
    for atom in observation.sequence:
        print(f'\t{atom}')


def atoms_to_sheet(dt: Program | Observation | Group) -> None:
    """
    Print out the atoms in a program or observation to a spreadsheet.
    """

    wb = Workbook()
    ws = wb.active
    ws.append(['id', 'exec_time', 'prog_time', 'part_time', 'observed', 'qa_state', 'guide_state'])

    def save_to_sheet(curr_atom: Atom):
        ws.cell(column=1, row=curr_atom.id + 1, value=curr_atom.id)
        ws.cell(column=2, row=curr_atom.id + 1, value=curr_atom.exec_time.total_seconds())
        ws.cell(column=3, row=curr_atom.id + 1, value=curr_atom.prog_time.total_seconds())
        ws.cell(column=4, row=curr_atom.id + 1, value=curr_atom.part_time.total_seconds())
        ws.cell(column=5, row=curr_atom.id + 1, value=curr_atom.observed)
        ws.cell(column=6, row=curr_atom.id + 1, value=curr_atom.qa_state.value)
        ws.cell(column=7, row=curr_atom.id + 1, value=curr_atom.guide_state)

    # TODO: Output for larger formats(e.g Program) not required but might be good to have.
    if isinstance(dt, Program):
        for obs in dt.observations():
            if obs.obs_class in [ObservationClass.SCIENCE, ObservationClass.PARTNERCAL]:
                for atom in obs.sequence:
                    ws.title = f'{obs.id}'
                    save_to_sheet(atom)
                ws = wb.create_sheet()
                ws.append(['id', 'exec_time', 'prog_time', 'part_time', 'observed', 'qa_state', 'guide_state'])
        wb.save(f'{dt.id}.xlsx')

    elif isinstance(dt, Observation):
        print(f'Observation: {dt.id}')
        for atom in dt.sequence:
            save_to_sheet(atom)
        wb.save(f'{dt.id}.xlsx')
    elif isinstance(dt, Group):
        raise NotImplementedError
    else:
        raise ValueError(f'Unsupported type: {type(dt)}')


def print_plans(all_plans: List[Plans]) -> None:
    """
    Print out the visit plans.
    """

    for plans in all_plans:
        print(f'\n\n+++++ NIGHT {plans.night + 1} +++++')
        for plan in plans:
            print(f'Plan for site: {plan.site.name}')
            for visit in plan.visits:
                print(f'\t{visit.start_time}   {visit.obs_id.id:20} {visit.score:8.2f}')
