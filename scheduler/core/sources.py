from enum import Enum

from scheduler.services.resource import OcsResourceService
from scheduler.services.environment import Env

class Services(Enum):
    ENV = 'env'
    RESOURCE = 'resource'
    CHRONICLE = 'chronicle'

class Origins(Enum):
    FILE = 'file'
    OCS = 'ocs'
    GPP = 'gpp'


class Sources:
    """
    Sources provide the scheduler with the correct source info for each service.
    Default should be GPP connections. Other modes are OCS mock services and custom files.
    """
    def __init__(self):
        self.origin = None
        self.env = None
        self.resource = None
        self.chronicle = None

    def set_origin(self, origin: Origins):
        # All sources must be Singleton's
        match origin:
            case Origins.OCS:
                self.origin = origin
                self.env = Env()
                self.resource = OcsResourceService()
                self.chronicle = None # Pending development
            case Origins.GPP:
                raise RuntimeError('GPP source not implemented yet!')

    def use_file(self, service: Services, file):
        match service:
            case Services.ENV:
                # Weather faults?
                pass
            case Services.RESOURCE:
                pass
                # Calendar,
                # GMOS conf,
            case Services.CHRONICLE:
                # Faults
                # Task
                pass

