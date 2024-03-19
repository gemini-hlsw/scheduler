# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from enum import Enum
from typing import final, NoReturn, Optional

from scheduler.services.abstract import ExternalService
from scheduler.services.environment import OcsEnvService
from scheduler.services.resource import OcsResourceService

from lucupy.types import Instantiable


__all__ = [
    'Origin',
    'OcsOrigin',
    'GppOrigin',
    'Origins',
]


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

    @staticmethod
    def _load_from_pickle():
        import os
        files_in_current_directory = os.listdir('./scheduler/services/resource/')
        pickle_files = [file for file in files_in_current_directory if file.endswith('.pickle')]
        are_pickle_files_present = len(pickle_files) > 0

        if are_pickle_files_present:
            with open('./scheduler/services/resource/resource.pickle', 'rb') as f:
                resource = pickle.load(f)
                return resource
        else:
            return None

    def load(self) -> OcsOrigin:
        if not self.is_loaded:
            self.resource = self._load_from_pickle() or OcsResourceService()
            self.resource.setup()
            self.env = OcsEnvService()
            self.is_loaded = True
            return self
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
