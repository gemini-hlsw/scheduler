import os
from api.ocs import OcsProgramProvider
from common.minimodel import NodeGroup, Observation

from typing import NoReturn

if __name__ == '__main__':
    provider = OcsProgramProvider(os.path.join('..', 'data', 'programs.json'))
    data = provider.load_program(os.path.join('data', 'GN-2018B-Q-101.json'))

    program = OcsProgramProvider.parse_program(data['PROGRAM_BASIC'])
    print(f'Program: {program.id}')

    def sep(depth: int) -> str:
        return '----- ' * depth

    def print_observation(depth: int, obs: Observation) -> NoReturn:
        print(f'{sep(depth)} Observation: {obs.id}')
        for atom in obs.sequence:
            print(f'{sep(depth + 1)} {atom}')

    def print_group(depth: int, group: NodeGroup) -> NoReturn:
        print(f'{sep(depth)} Group: {group.id}')
        for child in group.children:
            # Is this a subgroup or an observation?
            if isinstance(child, NodeGroup):
                print_group(depth + 1, child)
            elif isinstance(child, Observation):
                print_observation(depth + 1, child)

    # Print the group and atom information.
    print_group(program.root_group)