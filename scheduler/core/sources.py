from enum import Enum
from abc import ABC, abstractmethod
from typing import ClassVar, Optional, NoReturn
from scheduler.services.resource import OcsResourceService
from scheduler.services.environment import Env

class Services(Enum):
    ENV = 'env'
    RESOURCE = 'resource'
    CHRONICLE = 'chronicle'

class Origin(ABC):

    resource: ClassVar[Optional[OcsResourceService]] = None
    env: ClassVar[Optional[Env]] = None
    chronicle = None # Missing typing for lack of implementation
    is_loaded: ClassVar[bool] = False

    @abstractmethod
    def load(self) -> NoReturn:
        raise NotImplementedError('load Origin')


    def __str__(self):
        return self.__class__.__name__.replace('Origin','').upper()


class OCSOrigin(Origin):

    def load(self) -> 'OCSOrigin':
        if not OCSOrigin.is_loaded:
            OCSOrigin.resource = OcsResourceService()
            OCSOrigin.env = Env()
            # OCSOrigin.chronicle
            OCSOrigin.is_loaded = True
            return self


class GPPOrigin(Origin):
    def load(self) -> NoReturn:
        raise NotImplementedError('GPP sources are not implemented')

class FileOrigin(Origin):
    def load(self):
        pass

class Origins(Enum):
    FILE = FileOrigin()
    OCS = OCSOrigin()
    GPP = GPPOrigin()



class Sources:
    """
    Sources provide the scheduler with the correct source info for each service.
    Default should be GPP connections. Other modes are OCS mock services and custom files.
    """

    def __init__(self, origin: Origin = Origins.OCS.value):
        self.origin = origin.load() if origin else None

    def set_origin(self, origin: Origin):
        self.origin = origin.load()

    def use_file(self, service: Services, files):
        match service:
            case Services.ENV:
                # Weather faults?
                pass
            case Services.RESOURCE:

                pass
                # Calendar, self._load_spreadsheet()
                # GMOS conf,
            case Services.CHRONICLE:
                # Faults
                # Task
                pass

