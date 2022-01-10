#!/usr/bin/python3

import logging
import os
import zipfile

import requests
import telnetlib
from typing import NoReturn, FrozenSet
from zipfile import ZipFile

from common.minimodel import ProgramTypes, Site, SemesterHalf


# The default program types to download from the ODB.
DEFAULT_PROGRAM_TYPES = frozenset([
    ProgramTypes.Q,
    # ProgramTypes.C,
    ProgramTypes.FT,
    ProgramTypes.DD,
    # ProgramTypes.DS,
    # ProgramTypes.SV,
    ProgramTypes.LP]
)

# DEFAULT_SITES = frozenset([
#     Site.GN,
#     Site.GS
# ])
#
# DEFAULT_SEMESTERS = frozenmap([
#     f'2018{SemesterHalf.A.value}',
#     f'2018{}'
# ])


def download_programs(server: str = 'gnodbscheduler',
                      telnet_port: int = 8224,
                      read_port: int = 8442,
                      output_path: str = 'programs',
                      zip_file: str = 'programs.zip',
                      program_types: FrozenSet[ProgramTypes] = DEFAULT_PROGRAM_TYPES) -> NoReturn:
    """
    Download a list of the program types of interest
    """
    program_codes = frozenset([e.value.abbreviation for e in program_types])

    with telnetlib.Telnet(server, telnet_port) as tn:
        # Get a list of all programs in the ODB.
        tn.read_until(b'g! ')
        tn.write('lsprogs'.encode('ascii') + b'\n')
        program_names = [name.decode('ascii') for name in tn.read_until(b'g! ').split()]

        # Filter based on program type.
        # We are interested in programs of the form Gs-yyyys-T-nnn
        # where T is the program type.
        filtered_programs = set()
        for program_name in program_names:
            program_info = program_name.split('-')
            if len(program_info) != 4:
                logging.info(f'Skipping {program_name}: not a recognized program')
            elif program_info[2] not in program_codes:
                logging.info(f'Skipping {program_name}: {program_info[2]} is not a selected program type')
            else:
                filtered_programs.add(program_name)

        # Now download all the programs.
        if not os.path.isdir(output_path):
            os.mkdir(output_path)

        os.chdir(output_path)
        downloaded_programs = set()
        for program_name in list(filtered_programs)[:10]:
            output_file = f'{program_name}.json'
            params = {'id': program_name}
            r = requests.get(f'http://{server}:{read_port}/programexport', params)
            with open(output_file, 'w') as f:
                if r.status_code == 200:
                    f.write(r.text)
                    downloaded_programs.add(output_file)
                else:
                    logging.warning(f'Could not retrieve program {program_name}, status code {r.status_code}')

        if os.path.isfile(zip_file):
            os.unlink(zip_file)

        with ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for program_file in downloaded_programs:
                zf.write(program_file)

                # Remove the program file as we are done with it.
                os.unlink(program_file)


if __name__ == '__main__':
    download_programs()
