# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import os
import logging

from lucupy.minimodel import ALL_SITES
from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties

from definitions import ROOT_DIR
from scheduler.core.builder.blueprint import CollectorBlueprint
from scheduler.core.builder.builder import SchedulerBuilder
from scheduler.core.components.collector import *
from scheduler.core.output import print_collector_info
from scheduler.core.programprovider.ocs import read_ocs_zipfile, OcsProgramProvider
from scheduler.services import logger_factory
from scheduler.services.resource.filters import *
from scheduler.core.resourcemanager import ResourceManager

# This is a demo or QA testing ground for the filter functionality.
if __name__ == '__main__':
    logger = logger_factory.create_logger(__name__, logging.INFO)
    ObservatoryProperties.set_properties(GeminiProperties)

    # Configure and build the components.
    builder = SchedulerBuilder()

    # Read in a list of JSON data
    programs = read_ocs_zipfile(os.path.join(ROOT_DIR, 'scheduler', 'data', '2018B_program_samples.zip'))

    collector_blueprint = CollectorBlueprint(
        ['2018B'],
        ['SCIENCE', 'PROGCAL', 'PARTNERCAL'],
        ['Q', 'LP', 'FT', 'DD'],
        1.0
    )

    collector = SchedulerBuilder.build_collector(
        start=Time("2018-10-01 08:00:00", format='iso', scale='utc'),
        end=Time("2018-10-03 08:00:00", format='iso', scale='utc'),
        sites=ALL_SITES,
        blueprint=collector_blueprint
    )
    # Create the Collector and load the programs.
    collector.load_programs(program_provider_class=OcsProgramProvider,
                            data=programs)

    # Output the state of and information calculated by the Collector.
    print_collector_info(collector, samples=60)

    # Store the programs from the collector.
    program_data = collector._programs # noqa

    # Get the resource manager.
    resource_manager = ResourceManager()

    # Example filters.
    # Print the resources from each observation.
    print('\n\n\nProgram Information:')
    for pid, program in program_data.items():
        print(f'ProgramID: {pid}')
        root_group = program.root_group
        for group in root_group.children:
            print(f'\tGroupID: {group.unique_id()}, resources needed: f{group.required_resources()}')

    # ProgramPermissionFilter:
    program_ids = frozenset({
        'GS-2018B-Q-103',
        'GS-2018B-Q-105'
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
            resource_manager.lookup_resource('GMOS-S'),
            resource_manager.lookup_resource('Mirror'),
            resource_manager.lookup_resource('B600'),
            resource_manager.lookup_resource('10005374'),
            resource_manager.lookup_resource('10000009')
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
                print(f'+++ Group {group.unique_id()} is accepted: '
                      f'{", ".join(r.id for r in group.required_resources())} available.')
            else:
                print(f'--- Group {group.unique_id()} is rejected: '
                      f'{", ".join(r.id for r in group.required_resources() - resources)} unavailable.')

    priority_program_ids = frozenset({
        'GS-2018B-Q-103'
    })
    f_program_priority = ProgramPriorityFilter(
        program_ids=priority_program_ids
    )

    # Apply the filter.
    print(f'\n\nFILTERING ON PRIORITY FOR {priority_program_ids}:')
    for pid, program in Collector._programs.items(): # noqa
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
                    print(f'\t+++ Group {group.unique_id()} is accepted (resources available).')
                else:
                    print(f'\t--- Group {group.unique_id()} is rejected.')
        else:
            print(f'--- Program {pid} is rejected.')
