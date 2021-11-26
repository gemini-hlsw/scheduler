from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, IntEnum, auto
import numpy as np
import numpy.typing as npt
from typing import List, Mapping, Optional, Set, Union


# NOTE: In order to use numpy.typing, this file requires the 1.21 numpy package to be installed via:
# pip install numpy==1.21.4


class Site(str, Enum):
    """
    This will have to be customized by a given observatory if used independently
    of Gemini.
    """
    GN = 'Gemini North'
    GS = 'Gemini South'


@dataclass(unsafe_hash=True)
class ObservingPeriod:
    """
    This class represents a period under observation and contains visibility
    and scoring calculations for a night.
    """
    start: datetime
    vishours: float
    airmass: npt.NDArray[np.float]
    hour_angle: npt.NDArray[np.float]
    alt: npt.NDArray[np.float]
    az: npt.NDArray[np.float]
    parallactic_angle: npt.NDArray[np.float]
    sbcond: npt.NDArray[np.float]
    visfrac: npt.NDArray[np.float]
    score: Optional[npt.NDArray[np.float]] = None


class TimeAccountingCode(str, Enum):
    """
    This will have to be customized for a given observatory if used independently
    of Gemini.
    """
    AR = 'Argentina'
    AU = 'Australia'
    BR = 'Brazil'
    CA = 'Canada'
    CFH = 'CFHT Exchange'
    CL = 'Chile'
    KR = 'Republic of Korea'
    DD = "Director's Time"
    DS = 'Demo Science'
    GS = 'Gemini Staff'
    GT = 'Guaranteed Time'
    JP = 'Subaru'
    LP = 'Large Program'
    LTP = 'Limited-term Participant'
    SV = 'System Verification'
    UH = 'University of Hawaii'
    UK = 'United Kingdom'
    US = 'United States'
    XCHK = 'Keck Exchange'


@dataclass
class TimeAllocation:
    category: TimeAccountingCode
    program_awarded: timedelta
    partner_awarded: timedelta
    program_used: timedelta
    partner_used: timedelta

    def total_awarded(self) -> timedelta:
        return self.program_awarded + self.partner_awarded

    def total_used(self) -> timedelta:
        return self.program_used + self.partner_used


@dataclass(frozen=True)
class TimingWindow:
    """
    Representation of timing windows in the mini-model.

    For infinite duration, set duration to timedelta.max.
    For repeat, -1 means forever repeating, 0 means non-repeating.
    For period, None should be used if repeat < 1.
    """
    start: datetime
    duration: timedelta
    repeat: int
    period: Optional[timedelta]


class SkyBackground(float, Enum):
    SB20 = 0.2
    SB50 = 0.5
    SB80 = 0.8
    SBANY = 1.0


class CloudCover(float, Enum):
    CC50 = 0.5
    CC70 = 0.7
    CC80 = 0.8
    CCANY = 1.0


class ImageQuality(float, Enum):
    IQ20 = 0.2
    IQ70 = 0.7
    IQ85 = 0.85
    IQANY = 1.0


class WaterVapor(float, Enum):
    WV20 = 0.2
    WV50 = 0.5
    WV80 = 0.8
    WVANY = 1.0


class Stehl(float, Enum):
    S00 = 0.0
    S02 = 0.2
    S04 = 0.4
    S06 = 0.6
    S08 = 0.8
    S10 = 1.0


class ElevationType(IntEnum):
    HOUR_ANGLE = auto()
    AIRMASS = auto()


@dataclass(frozen=True)
class Constraints:
    cc: CloudCover
    iq: ImageQuality
    sb: SkyBackground
    wv: WaterVapor
    # constrast: Constrast
    elevation_type: ElevationType
    elevation_min: float
    elevation_max: float
    timing_windows: List[TimingWindow]
    # clearance_windows: Optional[List[ClearanceWindow]] = None
    strehl: Optional[Stehl] = None


class MagnitudeSystem(Enum):
    VEGA = auto()
    AB = auto()
    JY = auto()


@dataclass(frozen=True)
class MagnitudeBand:
    """
    Values for center and width are specified in microns.
    """
    name: str
    center: float
    width: float
    system: MagnitudeSystem = MagnitudeSystem.VEGA
    description: Optional[str] = None


class MagnitudeBands(MagnitudeBand, Enum):
    """
    It is unconventional to use lowercase characters in an enum, but to differentiate
    them from the uppercase magnitude bands, we must.
    """
    u = MagnitudeBand('u', 0.356, 0.046, MagnitudeSystem.AB, 'UV')
    g = MagnitudeBand('g', 0.483, 0.099, MagnitudeSystem.AB, 'green')
    r = MagnitudeBand('r', 0.626, 0.096, MagnitudeSystem.AB, 'red')
    i = MagnitudeBand('i', 0.767, 0.106, MagnitudeSystem.AB, 'far red')
    z = MagnitudeBand('z', 0.910, 0.125, MagnitudeSystem.AB, 'near-infrared')
    U = MagnitudeBand('U', 0.360, 0.075, description='ultraviolet')
    B = MagnitudeBand('B', 0.440, 0.090, description='blue')
    V = MagnitudeBand('V', 0.550, 0.085, description='visual')
    UC = MagnitudeBand('UC', 0.610, 0.063, description='UCAC')
    R = MagnitudeBand('R', 0.670, 0.100, description='red')
    I = MagnitudeBand('I', 0.870, 0.100, description='infrared')
    Y = MagnitudeBand('Y', 1.020, 0.120)
    J = MagnitudeBand('J', 1.250, 0.240)
    H = MagnitudeBand('H', 1.650, 0.300)
    K = MagnitudeBand('K', 2.200, 0.410)
    L = MagnitudeBand('L', 3.760, 0.700)
    M = MagnitudeBand('M', 4.770, 0.240)
    N = MagnitudeBand('N', 10.470, 5.230)
    Q = MagnitudeBand('Q', 20.130, 1.650)
    AP = MagnitudeBand('AP', 0.550, 0.085, MagnitudeSystem.VEGA, 'apparent')


@dataclass(frozen=True)
class Magnitude:
    band: MagnitudeBands
    value: float
    error: float


class TargetType(Enum):
    BASE = auto()
    USER = auto()
    BLIND_OFFSET = auto()
    OFF_AXIS = auto()
    TUNING_STAR = auto()
    GUIDING = auto()
    OTHER = auto()


class GuideSpeed(IntEnum):
    SLOW = auto()
    MEDIUM = auto()
    FAST = auto()


class TargetTag(Enum):
    COMET = auto()
    ASTEROID = auto()
    MAJOR_BODY = auto()


# TODO: Should this be frozen? If so, subclasses must be frozen.
@dataclass
class Target(ABC):
    """
    RA and Dec should be specified in decimal degrees.
    """
    name: str
    magnitudes: Set[Magnitude]
    type: TargetType

    def __post_init__(self):
        pass

    def guide_speed(self) -> GuideSpeed:
        """
        Calculate the guide speed for this target.
        """
        pass


@dataclass
class SiderealTarget(Target):
    """
    For a SiderealTarget, we have an RA and Dec and then proper motion information
    to calculate the exact position.

    Proper motion must be specified in milliarcseconds / year.
    Epoch must be the decimal year.
    """
    ra: float
    dec: float
    pm_ra: float
    pm_dec: float
    epoch: float

    def __post_init__(self):
        super().__post_init__()


@dataclass
class NonsiderealTarget(Target):
    """
    For a NonsiderealTarget, we have arrays of ephemerides to specify the position.
    """
    des: str
    tag: TargetTag
    ra: npt.NDArray[np.float]
    dec: npt.NDArray[np.float]

    def __post_init__(self):
        super().__post_init__()


@dataclass(unsafe_hash=True, frozen=True)
class Resource:
    id: str
    name: str
    description: Optional[str] = None


class QAState(Enum):
    UNKNOWN = auto()
    USABLE = auto()
    PASS = auto()
    FAIL = auto()


@dataclass
class Atom:
    """
    The wavelength must be specified in microns.
    """
    id: str
    exec_time: timedelta
    prog_time: timedelta
    part_time: timedelta
    observed: bool
    qa_state: QAState
    guide_state: bool
    required_resources: Set[Resource]
    wavelength: float


class ObservationStatus(Enum):
    NEW = auto()
    INCLUDED = auto()
    PROPOSED = auto()
    APPROVED = auto()
    FOR_REVIEW = auto()
    READY = auto()
    ONGOING = auto()
    OBSERVED = auto()


class Priority(IntEnum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()


class TooType(IntEnum):
    INTERRUPT = auto()
    RAPID = auto()
    STANDARD = auto()


class SetupTimeType(Enum):
    FULL = auto()
    REACQ = auto()
    NONE = auto()


class ObservationClass(Enum):
    SCIENCE = auto()
    PROG_CAL = auto()
    PARTNER_CAL = auto()
    ACQ = auto()
    ACQ_CAL = auto()


@dataclass
class Observation:
    id: str
    internal_id: str
    title: str
    site: Site
    status: ObservationStatus
    active: bool
    priority: Priority
    setuptime_type: SetupTimeType
    acq_overhead: timedelta
    obs_class: ObservationClass
    exec_time: timedelta
    program_used: timedelta
    partner_used: timedelta
    constraints: Constraints
    targets: List[Target]
    guide_stars: Mapping[Resource, Target]
    sequence: List[Atom]
    too_type: Optional[TooType] = None

    def total_used(self) -> timedelta:
        return self.program_used + self.partner_used

    def required_resources(self) -> Set[Resource]:
        return {r for a in self.sequence for r in a.required_resources}

    def wavelengths(self) -> Set[float]:
        return {c.wavelength for c in self.sequence}

    def constraints(self) -> Set[Constraints]:
        return {self.constraints}


# Since Python doesn't allow classes to self-reference, we have to make a basic group
# from which to subclass.
@dataclass
class Group(ABC):
    id: str
    group_name: str
    number_to_observe: int
    delay_min: timedelta
    delay_max: timedelta

    def __post_init__(self):
        pass

    @abstractmethod
    def required_resources(self) -> Set[Resource]:
        pass

    @abstractmethod
    def wavelengths(self) -> Set[float]:
        pass

    @abstractmethod
    def constraints(self) -> Set[Constraints]:
        pass


# And then a group that contains children.
@dataclass
class NodeGroup(Group, ABC):
    children: List[Union[Group, Observation]]

    def __post_init__(self):
        super().__post_init__()

    def required_resources(self) -> Set[Resource]:
        return {r for c in self.children for r in c.required_resources()}

    def wavelengths(self) -> Set[float]:
        return {w for c in self.children for w in c.wavelengths()}

    def constraints(self) -> Set[Constraints]:
        return {cs for c in self.children for cs in c.constraints()}


class AndOption(Enum):
    CONSEC_ORDERED = auto()
    CONSEC_ANYORDER = auto()
    NIGHT_ORDERED = auto()
    NIGHT_ANYORDER = auto()
    CUSTOM = auto()


@dataclass
class AndGroup(NodeGroup):
    group_option: AndOption
    previous: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()


@dataclass
class OrGroup(NodeGroup):
    def __post_init__(self):
        super().__post_init__()


class Band(IntEnum):
    BAND1 = 1
    BAND2 = 2
    BAND3 = 3
    BAND4 = 4


class ProgramMode(Enum):
    QUEUE = auto()
    CLASSICAL = auto()
    PV = auto()


@dataclass
class Program:
    id: str
    internal_id: str
    band: Band
    thesis: bool
    mode: ProgramMode
    start_time: datetime
    end_time: datetime
    allocated_time: Set[TimeAllocation]
    root_group: Group
    too_type: Optional[TooType] = None

    def program_awarded(self) -> timedelta:
        return sum(t.program_awarded for t in self.allocated_time)

    def program_used(self) -> timedelta:
        return sum(t.program_used for t in self.allocated_time)

    def partner_awarded(self) -> timedelta:
        return sum(t.partner_awarded for t in self.allocated_time)

    def partner_used(self) -> timedelta:
        return sum(t.partner_used for t in self.allocated_time)

    def total_awarded(self) -> timedelta:
        return sum(t.total_awarded() for t in self.allocated_time)

    def total_used(self) -> timedelta:
        return sum(t.total_used() for t in self.allocated_time)


@dataclass(frozen=True)
class Visit:
    start_time: datetime
    end_time: datetime
    observation_id: str
    first_atom_id: str
    last_atom_id: str
    comment: str
    setuptime_type: SetupTimeType


@dataclass
class Plan:
    scheduled_atoms: Mapping[Site, Mapping[ObservingPeriod, List[Visit]]]