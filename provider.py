import os
from api.ocs import OCSProgramProvider

provider = OCSProgramProvider(os.path.join('..', 'data', 'programs.json'))
json = provider.load_program(os.path.join('data', 'GN-2018B-Q-101.json'))

program = provider.parse_program(json['PROGRAM_BASIC'])
provider._calculate_too_type_for_obs(program)
print(f'Program: {program.id}')

for group in program.root_group.children:
    print(f'----- Group: {group.id}')
    for obs in group.children:
        print(f'----- ----- Observation: {obs.id}')
        for atom in obs.sequence:
            print(f'----- ----- ----- {atom}')
