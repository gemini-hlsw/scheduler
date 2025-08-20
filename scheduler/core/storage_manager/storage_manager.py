# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
import json
import zipfile
from os import PathLike
from pathlib import Path
from typing import Optional, FrozenSet, Iterable, Dict, List, Any, NoReturn

from gpp_client import GPPClient, GPPDirector
from lucupy.meta import Singleton
from lucupy.minimodel import ProgramID, Program, Site, ObservationClass

from definitions import ROOT_DIR
from scheduler.core.builder.blueprint import Blueprints
from scheduler.core.programprovider.gpp import GppProgramProvider
from scheduler.core.programprovider.ocs import OcsProgramProvider
from scheduler.core.sources import Sources, Origins
from scheduler.services import logger_factory
from scheduler.services.visibility import calculate_target_snapshot

import traceback

_logger = logger_factory.create_logger(__name__)

__all__ = ['StorageManager', 'storage_manager']


DEFAULT_OCS_DATA_PATH = Path(ROOT_DIR) / 'scheduler' / 'data' / 'programs.zip'
DEFAULT_OCS_PROGRAM_ID_PATH = Path(ROOT_DIR) / 'scheduler' / 'data' / 'program_ids.redis.txt'
# If the selection is None, default to this list
DEFAULT_OCS_PROGRAMS_PATH = Path(ROOT_DIR) / 'scheduler' / 'data' / 'program_ids.txt'
DEFAULT_GPP_PROGRAMS_PATH = Path(ROOT_DIR) / 'scheduler' / 'data' / 'gpp_program_ids.txt'

class StorageManager(metaclass=Singleton):

    def __init__(self):
        self._providers = {
            'GPP': GppProgramProvider,
            'OCS': OcsProgramProvider
        }

        self._programs: Dict[ProgramID, Program] = {}
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

    @staticmethod
    def _read_program_ids_file(path: Path) -> Optional[FrozenSet[str]]:
        """
        Read program ids files
        """
        try:
            with path.open('r') as file:
                id_frozenset = frozenset(line.strip() for line in file if line.strip() and line.strip()[0] != '#')
        except FileNotFoundError:
            # If the file does not exist, set id_frozenset to None
            id_frozenset = None
        return id_frozenset

    async def _gpp_data_retrieval(self):
        """ Generator for  gpp data"""
        try:
            programs = await self._gpp_director.scheduler.program.get_all()
            return programs
        except ValueError as e:
            _logger.error(f'Problem querying programs data: \n{e}.')

    async def _ocs_data_retrieval(self):
        id_frozenset = self._read_program_ids_file(DEFAULT_OCS_PROGRAM_ID_PATH)
        return self._read_ocs_zipfile(
            DEFAULT_OCS_DATA_PATH,
            id_frozenset
        )

    def _process_programs(
            self, programs: List[Dict[Any,Any]],
            provider_key: str,
            obs_classes: FrozenSet[ObservationClass],
            source: Sources
    ) -> None:
        """Process programs from a given provider and update internal collections.

        Attributes:
            programs (List[Dict[Any,Any]]): Serialized programs.
            provider_key (str): The key of the provider, either OCS or GPP.
            obs_classes (FrozenSet[ObservationClass]): Observation classes to be filtered in the config.
            source (Sources): Source selected for external services (e.g. Resource).

        """
        program_provider = self._providers[provider_key](obs_classes, source)

        # Count the number of parse failures.
        bad_program_count = 0
        for next_program in programs:
            try:
                if len(next_program.keys()) == 1:
                    # Extract the data from the JSON program. We do not need the top label.
                    next_data = next(iter(next_program.values()))
                else:
                    # This is a dictionary from the provider
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
                traceback.print_exc()
                bad_program_count += 1
                _logger.warning(f'Could not parse {provider_key} program: {e}')

        if bad_program_count:
            _logger.error(f'Could not parse {bad_program_count} programs for {provider_key}.')

    def get_programs(self, program_ids: FrozenSet[ProgramID], sites: FrozenSet[Site] ) -> Dict[ProgramID, Program]:
        str_sites = [s.name for s in sites]
        return {
            p_id: self._programs[p_id] for p_id in self._programs.keys()
            if p_id.id in program_ids and p_id.id[:2] in str_sites
        }

    def get_observations(self, obs_ids: FrozenSet[str]) -> Iterable[dict]:
        return [self._observations[obs_id] for obs_id in obs_ids if obs_id in self._observations]

    def get_targets(self, target_names: FrozenSet[str]) -> Iterable[dict]:
        return [self._targets[target_name] for target_name in target_names if target_name in self._targets]

    def load_ocs_default_list(self) -> FrozenSet[str]:
        id_frozenset = self._read_program_ids_file(DEFAULT_OCS_PROGRAMS_PATH)
        return id_frozenset

    def load_gpp_default_list(self) -> FrozenSet[str]:
        id_frozenset = self._read_program_ids_file(DEFAULT_GPP_PROGRAMS_PATH)
        return id_frozenset

    async def initialize(self):
        """Starting point for the storage manager.
        Brings both OCS and GPP data to the storage manager and store it to be retrieved later.
        This process is done in the background at the starting of the server.
        """

        # for GPP
        # Allows to tell the collector to not filter those observations by default.
        obs_classes = Blueprints.collector.obs_classes
        # This seems pointless now that the Scheduler would run by process and the process dictates
        source = Sources()
        source.set_origin(Origins.SIM())
        programs = await self._gpp_data_retrieval()  # Assuming this exists
        _logger.info(f'Total GPP {len(programs)} programs')
        self._process_programs(programs, 'GPP', obs_classes, source)
        _logger.info(f'Programs after GPP data retrieval: {len(self._programs.keys())}')

        programs = await self._ocs_data_retrieval()
        programs_list = [p for p in programs]
        _logger.info(f'Total OCS {len(programs_list)} programs')
        source.set_origin(Origins.OCS())
        self._process_programs(programs_list, 'OCS', obs_classes, source)
        _logger.info(f'Programs after OCS data retrieval: {len(self._programs.keys())}')

storage_manager = StorageManager()