import os
import json
from typing import NoReturn

from api.abstract import ProgramProvider
from api.ocs import OcsProgramProvider
from common.minimodel import Group, Observation, Program


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
