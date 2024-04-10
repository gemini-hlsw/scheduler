# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import logging
from astropy.time import Time

from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties
from lucupy.minimodel import SemesterHalf
from lucupy.minimodel.site import ALL_SITES
from lucupy.minimodel import NightIndex
from lucupy.minimodel.semester import Semester
from lucupy.minimodel import ProgramID

from scheduler.core.builder.blueprint import CollectorBlueprint
from scheduler.core.builder.validationbuilder import ValidationBuilder
from scheduler.core.components.collector import Collector
from scheduler.core.eventsqueue import EventQueue
from scheduler.core.output import print_collector_info
from scheduler.core.sources import Sources
from scheduler.services import logger_factory
from scheduler.services.resource import (CompositeFilter, OcsResourceService, ProgramPriorityFilter,
                                         ProgramPermissionFilter, ResourcesAvailableFilter, ResourceService)

# This is a demo or QA testing ground for the filter functionality.
if __name__ == '__main__':
    logger = logger_factory.create_logger(__name__, logging.INFO)
    ObservatoryProperties.set_properties(GeminiProperties)

    num_nights_to_schedule = 1
    night_indices = frozenset(NightIndex(idx) for idx in range(num_nights_to_schedule))
    queue = EventQueue(night_indices, ALL_SITES)
    builder = ValidationBuilder(Sources(), queue)

    # Create the Collector and load the programs.
    collector_blueprint = CollectorBlueprint(
        ['SCIENCE', 'PROGCAL', 'PARTNERCAL'],
        ['Q', 'LP', 'FT', 'DD'],
        1.0
    )
    collector = builder.build_collector(
        start=Time("2018-10-01 08:00:00", format='iso', scale='utc'),
        end=Time("2018-10-03 08:00:00", format='iso', scale='utc'),
        semesters=frozenset({Semester(2018, SemesterHalf.B)}),
        sites=ALL_SITES,
        blueprint=collector_blueprint
    )

    # Output the state of and information calculated by the Collector.
    print_collector_info(collector)

    # Store the programs from the collector.
    program_data = collector._programs  # noqa

    # Get the resource manager.
    resource_service = OcsResourceService()

    # Example filters.
    # Print the resources from each observation.
    print('\n\n\nProgram Information:')
    for pid, program in program_data.items():
        print(f'ProgramID: {pid}')
        root_group = program.root_group
        for group in root_group.children:
            print(f'\tGroupID: {group.unique_id}, resources needed: f{group.required_resources()}')

    # ProgramPermissionFilter:
    program_ids = frozenset({
        ProgramID('GS-2018B-Q-103'),
        ProgramID('GS-2018B-Q-105')
    })
    f_program_filtering = ProgramPermissionFilter(
        program_ids=program_ids
    )

    print(f'\n\nPROGRAM FILTERING ON {program_ids}:')
    for pid, program in program_data.items():
        if f_program_filtering.program_filter(program):
            print(f'+++ Program {pid} is accepted.')
        else:
            print(f'--- Program {pid} is rejected.')

    f_neg_program_filtering = CompositeFilter(
        negative_filters=frozenset({f_program_filtering})
    )
    print(f'\n\nASIDE: NEGATIVE PROGRAM FILTERING ON {program_ids}:')
    for pid, program in program_data.items():
        if f_neg_program_filtering.program_filter(program):
            print(f'+++ Program {pid} is accepted.')
        else:
            print(f'--- Program {pid} is rejected.')

    resources = frozenset({
        ResourceService.lookup_resource('Gemini South'),
        ResourceService.lookup_resource('GMOS-S'),
        ResourceService.lookup_resource('Mirror'),
        ResourceService.lookup_resource('B600'),
        ResourceService.lookup_resource('10005374'),
        ResourceService.lookup_resource('10000009')
    })
    f_resources_available = ResourcesAvailableFilter(
        resources=resources
    )

    # Apply the filter.
    print(f'\n\nRESOURCE FILTERING GROUPS ON {resources}:')
    for pid, program in program_data.items():
        program_message = False
        root_group = program.root_group
        for group in root_group.children:
            if f_resources_available.group_filter(group):
                print(f'+++ Group {group.unique_id} is accepted: '
                      f'{", ".join(r.id for r in group.required_resources())} available.')
            else:
                print(f'--- Group {group.unique_id} is rejected: '
                      f'{", ".join(r.id for r in group.required_resources() - resources)} unavailable.')

    priority_program_ids = frozenset({
        ProgramID('GS-2018B-Q-103')
    })
    f_program_priority = ProgramPriorityFilter(
        program_ids=priority_program_ids
    )

    # Apply the filter.
    print(f'\n\nFILTERING ON PRIORITY FOR {priority_program_ids}:')
    for pid, program in Collector._programs.items():  # noqa
        if f_program_priority.program_priority_filter(program):
            print(f'+++ Program {pid} is high priority.')
        else:
            print(f'--- Program {pid} is normal priority.')

    f_final = CompositeFilter(
        positive_filters=frozenset({f_program_priority, f_resources_available, f_program_filtering})
    )

    print('\n\nCOMBINATION FILTER: programs, program_priority, resources available:')
    for pid, program in program_data.items():
        if f_final.program_filter(program):
            if f_final.program_priority_filter(program):
                print(f'+++ Program {pid} is accepted (HIGH priority).')
            else:
                print(f'+++ Program {pid} is accepted (normal priority).')
            root_group = program.root_group
            for group in root_group.children:
                if f_final.group_filter(group):
                    print(f'\t+++ Group {group.unique_id} is accepted (resources available).')
                else:
                    print(f'\t--- Group {group.unique_id} is rejected.')
        else:
            print(f'--- Program {pid} is rejected.')
