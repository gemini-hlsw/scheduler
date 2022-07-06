# This has to be outside of Observation to avoid circular imports.
from enum import Enum


class ObservationMode(str, Enum):
    """
    TODO: This is not stored anywhere and is only used temporarily in the atom code in the
    TODO: OcsProgramExtractor. Should it be stored anywhere or is it only used in intermediate
    TODO: calculations? It seems to depend on the instrument and FPU.
    """
    UNKNOWN = 'unknown'
    IMAGING = 'imaging'
    LONGSLIT = 'longslit'
    IFU = 'ifu'
    MOS = 'mos'
    XD = 'xd'
    CORON = 'coron'
    NRM = 'nrm'