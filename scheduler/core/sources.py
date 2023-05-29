# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from enum import Enum
from abc import ABC, abstractmethod
from typing import ClassVar, Optional, NoReturn, Tuple
from scheduler.services.resource import OcsResourceService, FileResourceService
from scheduler.core.resourcemanager import ExternalService
from scheduler.services.environment import Env
from io import BytesIO

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
        return self.__class__.__name__.replace('Origin','').upper()


class OCSOrigin(Origin):

    def load(self) -> 'OCSOrigin':
        if not self.is_loaded:
            self.resource = OcsResourceService()
            self.env = Env()
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

class Origins(Enum):
    FILE = FileOrigin()
    OCS = OCSOrigin()
    GPP = GPPOrigin()



class Sources:
    """
    Sources provide the scheduler with the correct source info for each service.
    Default should be GPP connections. Other modes are OCS services and custom files.
    """

    def __init__(self, origin: Origin = Origins.OCS.value):
        self.set_origin(origin)

    def set_origin(self, origin: Origin):
        self.origin = origin.load()

    def use_file(self,
                 service: Services,
                 calendar: BytesIO,
                 gmos_fpu: BytesIO,
                 gmos_gratings: BytesIO) -> Tuple[bool, str]:

        match service:
            case Services.ENV:
                # Weather faults?
                return False, 'Handler not implemented yet!'

            case Services.RESOURCE:
                # Check that the amount of files is correct
                if calendar and gmos_fpu and gmos_gratings:
                    file_resource = FileResourceService()

                    for sites in files_input.sites:
                        suffix = ('s' if site == Site.GS else 'n').upper()
                        file_resource.load_files(f'GMOS{suffix}_fpu_barcode.txt',
                                                 gmos_fpu,
                                                 gmos_gratings,
                                                 calendar)

                    self.set_origin(Origin.FILE.value)
                    self.origin.resource = file_resource
                    return True, 'Resource files correctly loaded!'

                else:
                    return False, 'Missing files to load!'
            case Services.CHRONICLE:
                # Faults
                # Task
                return False, 'Handler not implemented yet!'

