# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from gpp_client import GPPClient, GPPDirector
from lucupy.meta import Singleton


from scheduler.core.builder.blueprint import Blueprints
from scheduler.core.programprovider.gpp import GppProgramProvider
from scheduler.core.programprovider.ocs import OcsProgramProvider
from scheduler.core.sources import Sources, Origins
from scheduler.services import logger_factory

_logger = logger_factory.create_logger(__name__)

__all__ = ['InformationStorage', 'info_storage']

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


    async def _gpp_data_retrieval(self):
        """ Generator for  gpp data"""

        try:
            programs = await self._gpp_director.scheduler.program.get_all()
            return programs
        except ValueError as e:
            _logger.error(f'Problem querying programs data: \n{e}.')

    async def _ocs_data_retrieval(self):
        pass


    def load_programs(self):
        pass

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


info_storage = InformationStorage()