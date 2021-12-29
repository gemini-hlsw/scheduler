from dataclasses import dataclass

@dataclass
class ProgramKeys:
    ID = 'programId'
    INTERNAL_ID = 'key'
    BAND = 'queueBand'
    THESIS = 'isThesis'
    MODE = 'programMode'
    TOO_TYPE = 'tooType'
    NOTE = 'INFO_SCHEDNOTE'

@dataclass
class NoteKeys:
    TITLE = 'title'
    TEXT = 'text'
    KEY = 'key'

@dataclass
class TAKeys:
    CATEGORIES: str = 'timeAccountAllocationCategories'
    CATEGORY: str = 'category'
    AWARDED_TIME: str = 'awardedTime'
    PROGRAM_TIME: str = 'programTime'
    PARTNER_TIME: str = 'partnerTime'

@dataclass
class GroupsKeys:
    KEY = 'GROUP_GROUP_SCHEDULING'
    ORGANIZATIONAL_FOLDER = 'ORGANIZATIONAL_FOLDER'

@dataclass
class ObsKeys:
    KEY = 'OBSERVATION_BASIC'
    ID = 'observationId'
    INTERNAL_ID = 'key'
    QASTATE = 'qaState'
    LOG = 'obsLog'
    STATUS = 'obsStatus'
    PRIORITY = 'priority'
    TITLE = 'title'
    SEQUENCE = 'sequence'
    SETUPTIME_TYPE = 'setupTimeType'
    SETUPTIME = 'setupTime'
    OBS_CLASS = 'obsClass'
    PHASE2 = 'phase2Status'
    TOO_OVERRIDE_RAPID = 'tooOverrideRapid'

@dataclass
class TargetKeys:
    KEY = 'TELESCOPE_TARGETENV'
    BASE = 'base'
    TYPE = 'type'
    RA = 'ra'
    DEC = 'dec'
    DELTARA = 'deltara'
    DELTADEC = 'deltadec'
    EPOCH = 'epoch'
    DES = 'des'
    TAG = 'tag'
    MAGNITUDES = 'magnitudes'
    NAME = 'name'

@dataclass
class ConstraintsKeys:
    KEY = 'SCHEDULING_CONDITIONS'
    CC  = 'cc'
    IQ = 'iq'
    SB = 'sb'
    WV = 'wv'
    ELEVATION_TYPE = 'elevationConstraintType'
    ELEVATION_MIN = 'elevationConstraintMin'
    ELEVATION_MAX = 'elevationConstraintMax'
    TIMING_WINDOWS = 'timingWindows'

@dataclass
class AtomKeys:
    OBS_CLASS = 'observe:class'
    INSTRUMENT = 'instrument:instrument'
    WAVELENGTH = 'instrument:observingWavelength'
    OBSERVED = 'metadata:complete'
    TOTAL_TIME = 'totalTime'
    OFFSET_P = 'telescope:p'
    OFFSET_Q = 'telescope:q'

@dataclass
class TimingWindowsKeys:
    TIMING_WINDOWS = 'timingWindows'
    START = 'start'
    DURATION = 'duration'
    REPEAT = 'repeat'
    PERIOD = 'period'

@dataclass
class MagnitudeKeys:
    NAME = 'name'
    VALUE = 'value'
