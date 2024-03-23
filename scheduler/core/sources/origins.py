# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import final, Final, NoReturn, Optional
import pickle

from lucupy.types import Instantiable

from definitions import ROOT_DIR
from scheduler.services.abstract import ExternalService
from scheduler.services.logger_factory import create_logger
from scheduler.services.environment import OcsEnvService
from scheduler.services.resource import OcsResourceService


__all__ = [
    'Origin',
    'OcsOrigin',
    'GppOrigin',
    'Origins',
]

logger = create_logger(__name__)


class Origin(ABC):
    def __init__(self,
                 resource: Optional[ExternalService] = None,
                 env: Optional[ExternalService] = None,
                 is_loaded: bool = False):
        self.resource = resource
        self.env = env
        self.is_loaded = is_loaded

    @abstractmethod
    def load(self) -> NoReturn:
        raise NotImplementedError('load Origin')

    def __str__(self):
        return self.__class__.__name__.replace('Origin', '').upper()


@final
class OcsOrigin(Origin):
    _env_path: Final[Path] = Path(ROOT_DIR) / 'scheduler' / 'pickles' / 'ocsenv.pickle'
    _resource_path: Final[Path] = Path(ROOT_DIR) / 'scheduler' / 'pickles' / 'ocsresource.pickle'

    def load(self) -> OcsOrigin:
        if not self.is_loaded:
            try:
                with open(OcsOrigin._resource_path, 'rb') as res_pickle:
                    self.resource = pickle.load(res_pickle)
                    logger.info('Read OCS Resource service from pickle.')
            except Exception:
                logger.info('Creating and pickling OCS Resource service.')
                self.resource = OcsResourceService()
                OcsOrigin._resource_path.parent.mkdir(parents=True, exist_ok=True)
                with open(OcsOrigin._resource_path, 'wb') as res_pickle:
                    pickle.dump(self.resource, res_pickle)

            try:
                with open(OcsOrigin._env_path, 'rb') as res_env:
                    self.env = pickle.load(res_env)
                    logger.info('Read OCS Env service from pickle.')
            except Exception:
                logger.info('Creating and pickling OCS Env service.')
                self.env = OcsEnvService()
                OcsOrigin._env_path.parent.mkdir(parents=True, exist_ok=True)
                with open(OcsOrigin._env_path, 'wb') as env_pickle:
                    pickle.dump(self.env, env_pickle)

        self.is_loaded = True
        return self


@final
class GppOrigin(Origin):
    def load(self) -> NoReturn:
        raise NotImplementedError('GPP sources are not implemented')


@final
class FileOrigin(Origin):
    def load(self):
        return self


@final
class Origins(Instantiable[Origin], Enum):
    FILE = Instantiable(lambda: FileOrigin())
    OCS = Instantiable(lambda: OcsOrigin())
    GPP = Instantiable(lambda: GppOrigin())
