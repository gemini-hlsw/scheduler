from enum import Enum
from abc import ABC, abstractmethod
from typing import ClassVar, Optional, NoReturn, Tuple
from scheduler.services.resource import OcsResourceService, FileResourceService
from scheduler.services.environment import Env
from io import BytesIO

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
        return self

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
                        # reutilize this file.
                        file_resource._load_fpu_to_barcodes(site, f'GMOS{suffix}_fpu_barcode.txt')
                        file_resource._load_csv(site,
                                                lambda r: {self._itcd_fpu_to_barcode[site][r[0].strip()].id} | {i.strip() for i in r[1:]},
                                                file=gmos_fpu)

                        file_resource._load_csv(site,
                                                lambda r: {'Mirror'} | {i.strip().replace('+', '') for i in r},
                                                file=gmos_gratings)
                    file_resource._load_spreadsheet(file=calendar)

                    self.set_origin(Origin.FILE.value)
                    self.origin.resource = file_resource
                    return True, 'Resource files correctly loaded!'

                else:
                    return False, 'Missing files to load!'
            case Services.CHRONICLE:
                # Faults
                # Task
                return False, 'Handler not implemented yet!'

