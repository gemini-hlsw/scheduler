# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
import json
import zipfile
from os import PathLike
from pathlib import Path
from typing import Optional, FrozenSet, Iterable

from gpp_client import GPPClient, GPPDirector
from lucupy.meta import Singleton

from definitions import ROOT_DIR
from scheduler.core.builder.blueprint import Blueprints
from scheduler.core.programprovider.gpp import GppProgramProvider
from scheduler.core.programprovider.ocs import OcsProgramProvider
from scheduler.core.sources import Sources, Origins
from scheduler.services import logger_factory
from scheduler.services.visibility import calculate_target_snapshot

_logger = logger_factory.create_logger(__name__)

__all__ = ['InformationStorage', 'info_storage']


DEFAULT_OCS_DATA_PATH = Path(ROOT_DIR) / 'scheduler' / 'data' / 'programs.zip'
DEFAULT_PROGRAM_ID_PATH = Path(ROOT_DIR) / 'scheduler' / 'data' / 'program_ids.redis.txt'

class InformationStorage(metaclass=Singleton):

    def __init__(self):
        self._providers = {
            'GPP': GppProgramProvider,
            'OCS': OcsProgramProvider
        }

        self._programs: dict = {}
        self._observations: dict = {}
        self._targets: dict = {}

        self._gpp_client = GPPClient()
        self._gpp_director = GPPDirector(self._gpp_client)

    @staticmethod
    def _read_ocs_zipfile(
            zip_file: str | PathLike[str],
            program_ids: Optional[FrozenSet[str]] = None
    ) -> Iterable[dict]:
        """
        Since for OCS we will use a collection of extracted ODB data, this is a
        convenience method to parse the data into a list of the JSON program data.
        """
        with zipfile.ZipFile(zip_file, 'r') as zf:
            for filename in zf.namelist():
                program_id = Path(filename).stem
                if program_ids is None or program_id in program_ids:
                    with zf.open(filename) as f:
                        contents = f.read().decode('utf-8')
                        _logger.debug(f'Adding program {program_id}.')
                        yield json.loads(contents)
                else:
                    _logger.debug(f'Skipping program {program_id} as it is not in the list.')


    async def _gpp_data_retrieval(self):
        """ Generator for  gpp data"""

        try:
            programs = await self._gpp_director.scheduler.program.get_all()
            return programs
        except ValueError as e:
            _logger.error(f'Problem querying programs data: \n{e}.')

    async def _ocs_data_retrieval(self):
        try:
            # Try to read the file and create a frozenset from its lines
            list_file = DEFAULT_PROGRAM_ID_PATH

            with list_file.open('r') as file:
                id_frozenset = frozenset(line.strip() for line in file if line.strip() and line.strip()[0] != '#')
        except FileNotFoundError:
            # If the file does not exist, set id_frozenset to None
            id_frozenset = None
        return self._read_ocs_zipfile(
            DEFAULT_OCS_DATA_PATH,
            id_frozenset
        )


    async def initialize(self):

        # for GPP
        # Allows to tell the collector to not filter those observations by default.
        obs_classes = Blueprints.collector.obs_classes
        # This seems pointless now that the Scheduler would run by process and the process dictates
        source = Sources().set_origin(Origins.SIM())
        programs = await self._gpp_data_retrieval()

        program_provider = self._providers['GPP'](obs_classes, source)
        try:
            for next_program in programs:
                if len(next_program.keys()) == 1:
                    # Extract the data from the OCS JSON program. We do not need the top label.
                    next_data = next(iter(next_program.values()))
                else:
                    # This is a dictionary from GPP
                    next_data = next_program

                program = program_provider.parse_program(next_data)
                if program is None:
                    continue

                if program.id in self._programs.keys():
                    _logger.warning(f'Data contains a repeated program with id {program.id} (overwriting).')

                self._programs[program.id] = program
                for obs in program.observations():
                    self._observations[obs.id] = obs
                    target = obs.base_target()
                    if target is not None:
                        self._targets[target.name] = target

        except Exception as e:
            _logger.warning(f'Could not parse program: {e}')


        programs = await self._ocs_data_retrieval()
        program_provider = self._providers['OCS'](obs_classes, source)
        try:
            for next_program in programs:
                if len(next_program.keys()) == 1:
                    # Extract the data from the OCS JSON program. We do not need the top label.
                    next_data = next(iter(next_program.values()))
                else:
                    # This is a dictionary from GPP
                    next_data = next_program

                program = program_provider.parse_program(next_data)
                if program is None:
                    continue

                if program.id in self._programs.keys():
                    _logger.warning(f'Data contains a repeated program with id {program.id} (overwriting).')

                self._programs[program.id] = program
                for obs in program.observations():
                    self._observations[obs.id] = obs
                    target = obs.base_target()
                    if target is not None:
                        self._targets[target.name] = target
        except Exception as e:
            _logger.warning(f'Could not parse program: {e}')


info_storage = InformationStorage()