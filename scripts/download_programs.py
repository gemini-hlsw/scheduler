import logging
from dataclasses import dataclass
import os
import requests
import telnetlib
import tempfile
from typing import Mapping, NoReturn, FrozenSet
import zipfile

from common.minimodel import ProgramTypes, Site, SemesterHalf

# Add level to the logger for more informative logging.
EXTRACT = logging.WARN


@dataclass(frozen=True)
class ODBServer:
    """
    Connection information for an OCS observing database (ODB) server.
    * name: name of the server to be used for the connection
    * telnet_port: the port at which the server accepts telnet connections to query program lists
    * read_port: the port at which we can perform http queries to retrieve program information
    """
    name: str
    telnet_port: int
    read_port: int


# The default server to use for the base.
DEFAULT_SERVER = ODBServer('gnodbscheduler', 8224, 8442)

# These are the semesters of interest that we will download from the ODB,
# organized by Site.
DEFAULT_SEMESTERS = {
    Site.GN: {f'2018{SemesterHalf.A.value}',
              f'2018{SemesterHalf.B.value}',
              f'2019{SemesterHalf.A.value}',
              f'2019{SemesterHalf.B.value}'},
    Site.GS: {f'2018{SemesterHalf.A.value}',
              f'2018{SemesterHalf.B.value}',
              f'2019{SemesterHalf.A.value}',
              f'2019{SemesterHalf.B.value}'}
}

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


def download_programs(server: ODBServer = DEFAULT_SERVER,
                      zip_file: str = os.path.join('..', 'data', 'programs.zip'),
                      semesters: Mapping[Site, str] = None,
                      program_types: FrozenSet[ProgramTypes] = DEFAULT_PROGRAM_TYPES) -> NoReturn:
    """
    Download a list of the programs from the specified server to a temporary directory and produce a zip file.
    * server: an observing database (ODB) server information
    * zip_file: the path of the zip file that will contain the programs
    * semesters: for each Site, the semesters of interest
    * program_types: the types of programs to download
    """
    # Avoid default mutable arguments.
    if semesters is None:
        semesters = DEFAULT_SEMESTERS

    program_codes = frozenset([e.value.abbreviation for e in program_types])
    sites = frozenset([s.name for s in semesters])

    with telnetlib.Telnet(server.name, server.telnet_port) as tn:
        # Get a list of all programs in the ODB.
        tn.read_until(b'g! ')
        tn.write('lsprogs'.encode('ascii') + b'\n')
        program_names = [name.decode('ascii') for name in tn.read_until(b'g! ').split()]

        # Filter based on program type.
        # We are interested in programs of the form Gs-yyyys-T-nnn
        # where T is the program type.
        filtered_programs = []
        for program_name in program_names:
            program_info = program_name.split('-')
            if len(program_info) != 4:
                logging.info(f'Skipping {program_name}: not a recognized program')
            elif program_info[0] not in sites:
                logging.info(f'Skipping {program_name}: {program_info[0]} is not a site of interest')
            elif program_info[1] not in semesters[Site[program_info[0]]]:
                logging.info(f'Skipping {program_name}: site {program_info[0]}, semester {program_info[1]} '
                             'is not a semester of interest')
            elif program_info[2] not in program_codes:
                logging.info(f'Skipping {program_name}: {program_info[2]} is not a selected program type')
            else:
                filtered_programs.append(program_name)

        # Sort for increased readability.
        filtered_programs.sort()
        logging.log(EXTRACT, f'Found: {len(filtered_programs)} programs.')

        # Download all the programs we have filtered.
        cwd = os.getcwd()
        output_path = tempfile.mkdtemp()
        if not os.path.isdir(output_path):
            os.mkdir(output_path)

        os.chdir(output_path)
        downloaded_programs = []

        for program_name in list(filtered_programs):
            logging.log(EXTRACT, f'Extracting {program_name}')
            output_file = f'{program_name}.json'
            params = {'id': program_name}
            r = requests.get(f'http://{server.name}:{server.read_port}/programexport', params)
            with open(output_file, 'w') as f:
                if r.status_code == 200:
                    f.write(r.text)
                    downloaded_programs.append(output_file)
                else:
                    logging.warning(f'Could not retrieve program {program_name}, status code {r.status_code}')

        zip_file = os.path.join(cwd, zip_file)
        if os.path.isfile(zip_file):
            os.unlink(zip_file)

        with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for program_file in downloaded_programs:
                zf.write(program_file)

                # Remove the program file as we are done with it.
                os.unlink(program_file)

        os.chdir(cwd)
        os.rmdir(output_path)


if __name__ == '__main__':
    # Uncomment to turn on the info logging.
    # logging.basicConfig(level=logging.INFO)
    logging.addLevelName(EXTRACT, "extract")
    download_programs()
