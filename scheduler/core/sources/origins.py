# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import final, NoReturn, Optional
import pickle

from definitions import ROOT_DIR
from scheduler.services.abstract import ExternalService
from scheduler.services.environment import OcsEnvService
from scheduler.services.resource import OcsResourceService

from lucupy.types import Instantiable

import time


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
    num_calls = 0

    def load(self) -> OcsOrigin:
        OcsOrigin.num_calls += 1
        print(f'Calls to OcsOrigin.load: {OcsOrigin.num_calls}.')
        pickle_resource_path = Path(ROOT_DIR) / 'resource.pickle'
        pickle_env_path = Path(ROOT_DIR) / 'env.pickle'
        print(f'OcsOrigin loading... is_loaded={self.is_loaded}')
        if not self.is_loaded:
            print('Starting OcsOrigin timer...')
            start_timer = time.perf_counter()
            if pickle_resource_path.exists():
                print(f'\tFound pickled resource service...')
                with open(pickle_resource_path, 'rb') as res_pickle:
                    self.resource = pickle.load(res_pickle)
            else:
                print(f'\tCreating resource service...')
                self.resource = OcsResourceService()
                print(f'\tPickling resource service...')
                with open(pickle_resource_path, 'wb') as res_pickle:
                    pickle.dump(self.resource, res_pickle)

            if pickle_env_path.exists():
                print(f'\tFound pickled env service...')
                with open(pickle_env_path, 'rb') as env_pickle:
                    self.env = pickle.load(env_pickle)
            else:
                print(f'\tCreating env service...')
                self.env = OcsEnvService()
                print(f'\tPickling env service...')
                with open(pickle_env_path, 'wb') as env_pickle:
                    pickle.dump(self.env, env_pickle)

            self.is_loaded = True
            end_timer = time.perf_counter()
            print(f'OcsOrigin setup: {(end_timer - start_timer):.2f} seconds')
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
