import os
import json
from typing import NoReturn

from api.ocs import OcsProgramProvider
from common.minimodel import Group, Observation

if __name__ == '__main__':
    provider = OcsProgramProvider
    # data = provider.load_program(os.path.join('data', 'GN-2018B-Q-101.json'))
    path = os.path.join('data', 'GN-2018B-Q-101.json')
    with open(path, 'r') as f:
        data = json.loads(f.read())

    program = OcsProgramProvider.parse_program(data['PROGRAM_BASIC'])
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
