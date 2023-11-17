# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from io import BytesIO
from typing import Optional, NoReturn

from scheduler.services.abstract import ExternalService
from scheduler.services.environment import OcsEnvService
from scheduler.services.resource import OcsResourceService, FileResourceService

from lucupy.minimodel import Site


# TODO: This file will need significant cleanup after the initial demo version is released.
class Services(Enum):
    ENV = 'env'
    RESOURCE = 'resource'
    CHRONICLE = 'chronicle'


class Origin(ABC):
    def __init__(self,
                 resource: Optional[ExternalService] = None,
                 env: Optional[ExternalService] = None,
                 chronicle: Optional[ExternalService] = None,
                 is_loaded: bool = False):
        self.resource = resource
        self.env = env
        self.chronicle = chronicle
        self.is_loaded = is_loaded

    @abstractmethod
    def load(self) -> NoReturn:
        raise NotImplementedError('load Origin')

    def __str__(self):
        return self.__class__.__name__.replace('Origin', '').upper()


class OCSOrigin(Origin):

    def load(self) -> OCSOrigin:
        if not self.is_loaded:
            self.resource = OcsResourceService()
            self.env = OcsEnvService()
            # OCSOrigin.chronicle
            self.is_loaded = True
            return self
        return self


class GPPOrigin(Origin):
    def load(self) -> NoReturn:
        raise NotImplementedError('GPP sources are not implemented')


class FileOrigin(Origin):
    def load(self):
        return self


class Instantiable:
    def __init__(self, func):
        self.func = func

    def __call__(self):
        return self.func()


class Origins(Enum):
    FILE = Instantiable(lambda: FileOrigin())
    OCS = Instantiable(lambda: OCSOrigin())
    GPP = Instantiable(lambda: GPPOrigin())


class Sources:
    """
    Sources provide the scheduler with the correct source info for each service.
    Default should be GPP connections. Other modes are OCS services and custom files.
    """

    def __init__(self, origin: Origin = Origins.OCS.value()):
        self.set_origin(origin)

    def set_origin(self, origin: Origin):
        self.origin = origin.load()

    def use_file(self,
                 files_input,
                 service: Services,
                 calendar: BytesIO,
                 gmos_fpu: BytesIO,
                 gmos_gratings: BytesIO) -> bool:

        match service:
            case Services.ENV:
                # Weather faults?
                return False

            case Services.RESOURCE:
                # Check that the amount of files is correct
                if calendar and gmos_fpu and gmos_gratings:
                    file_resource = FileResourceService()

                    # TODO: files_input is undefined. This will be fixed later.
                    for site in files_input.sites:
                        suffix = ('s' if site == Site.GS else 'n').upper()
                        file_resource.load_files(f'GMOS{suffix}_fpu_barcode.txt',
                                                 gmos_fpu,
                                                 gmos_gratings,
                                                 calendar)

                    self.set_origin(Origin.FILE.value)
                    self.origin.resource = file_resource
                    return True

                else:
                    raise ValueError('Missing files to load for service ')
            case Services.CHRONICLE:
                # Faults
                # Task
                return False
