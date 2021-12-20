# NOTE: In order to use numpy.typing, this file requires the 1.21 numpy package to be installed via:
# pip install numpy==1.21.4

import logging
from abc import ABC, abstractmethod
from astropy.coordinates import EarthLocation, UnknownSiteException
from astropy.time import Time
from astropy import units as u
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, IntEnum, auto
import numpy.typing as npt
from pytz import timezone, UnknownTimeZoneError
from typing import ClassVar, List, Mapping, Optional, Set, Union


class SiteInformation:
    def __init__(self,
                 name: str,
                 astropy_lookup: str = None):
        """
        AstroPy location lookups for Gemini North and South are of the form:
        * gemini_north
        * gemini_south
        This conversion will happen automatically if astropy_lookup is None.

        If necessary, other observatories should provide hard astropy_lookup values.

        Time zone information for a site is also included here.
        """
        if astropy_lookup is None:
            astropy_lookup = name.lower().replace(' ', '_')

        self.name = name

        try:
            self.location = EarthLocation.of_site(astropy_lookup)
        except UnknownSiteException:
            logging.error(f'Unknown site lookup: {astropy_lookup}')

        timezone_info = self.location.info.meta['timezone']
        try:
            self.time_zone = timezone(timezone_info)
        except UnknownTimeZoneError:
            logging.error(f'Unknown time zone lookup: {timezone_info}')


class Site(Enum):
    """
    This will have to be customized by a given observatory if used independently
    of Gemini.
    """
    GN = SiteInformation('Gemini North')
    GS = SiteInformation('Gemini South')


@dataclass
class SemesterHalf(Enum):
    A = 'A'
    B = 'B'


@dataclass
class Semester:
    year: int
    half: SemesterHalf


@dataclass(unsafe_hash=True)
class ObservingPeriod:
    """
    This class represents a period under observation and contains visibility
    and scoring calculations for a night.

    It contains the constants:
    * MAX_AIRMASS: the maximum possible value for airmass.
    * CLASSICAL_NIGHT_LENGTH: the timedelta length for a classical observing night.
    """
    start: datetime
    length: timedelta
    vishours: float
    airmass: npt.NDArray[float]
    hour_angle: npt.NDArray[float]
    alt: npt.NDArray[float]
    az: npt.NDArray[float]
    parallactic_angle: npt.NDArray[float]
    sbcond: npt.NDArray[float]
    visfrac: npt.NDArray[float]
    score: Optional[npt.NDArray[float]] = None

    MAX_AIRMASS: ClassVar[float] = 2.3
    CLASSICAL_NIGHT_LENGTH: ClassVar[timedelta] = timedelta(hours=10)


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

    INFINITE_DURATION: ClassVar[int] = timedelta.max
    FOREVER_REPEATING: ClassVar[int] = -1
    NON_REPEATING: ClassVar[int] = 0
    NO_PERIOD: ClassVar[Optional[timedelta]] = None

    # A number to be used by the Scheduler to represent infinite repeats from the
    # perspective of the OCS: if FOREVER_REPEATING is selected, then it is converted
    # into this for calculation purposes.
    _OCS_INFINITE_REPEATS: ClassVar[int] = 1000

    @property
    def ocs_infinite_repeats(self):
        return self._OCS_INFINITE_REPEATS


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


@dataclass
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

    # For performance increase to avoid repeated computation.
    # Divide a time in milliseconds by this to get the quantity in hours.]
    _MS_TO_H: ClassVar[int] = u.hour.to('ms') * u.hour

    def __post_init__(self):

        """
        Convert the timing window information to more natural units, i.e. a list of
         Time, which is for more convenient processing.

         This creates a property on the Constraints called ot_timing_windows.
        """
        self.ot_timing_windows = []

        # Collect the timing window information as arrays from the TimingWindow list.
        starts = (tw.start.timestamp() for tw in self.timing_windows)
        durations = (tw.duration.total_seconds() for tw in self.timing_windows)
        repeats = (tw.repeat for tw in self.timing_windows)
        periods = (tw.period for tw in self.timing_windows)

        for (start, duration, repeat, period) in zip(starts, durations, repeats, periods):
            t0 = float(start) * u.ms
            begin = Time(t0.to_value('s'), format='unix', scale='utc')
            duration = TimingWindow.INFINITE_DURATION if duration == -1 else duration / Constraints._MS_TO_H
            repeat = TimingWindow.ocs_infinite_repeats if repeat == TimingWindow.FOREVER_REPEATING else max(1, repeat)
            period = period / Constraints._MS_TO_H

            for i in range(repeat):
                window_start = begin + i * period
                window_end = window_start + duration
                self.timing_windows.append(Time[window_start, window_end])


class MagnitudeSystem(Enum):
    VEGA = auto()
    AB = auto()
    JY = auto()


@dataclass(frozen=True)
class MagnitudeBand:
    """
    Values for center and width are specified in microns.
    These should NOT be created: they are fully enumerated in MagnitudeBands, so
    they should be looked up by name there.
    """
    name: str
    center: float
    width: float
    system: MagnitudeSystem = MagnitudeSystem.VEGA
    description: Optional[str] = None


class MagnitudeBands(Enum):
    """
    It is unconventional to use lowercase characters in an enum, but to differentiate
    them from the uppercase magnitude bands, we must.

    Look up the MagnitudeBand from this Enum as follows:
    MagnitudeBands[name]
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
    AP = MagnitudeBand('AP', 0.550, 0.085, description='apparent')


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
    ra: npt.NDArray[float]
    dec: npt.NDArray[float]

    def __post_init__(self):
        super().__post_init__()


@dataclass(unsafe_hash=True, frozen=True)
class Resource:
    id: str
    name: str
    description: Optional[str] = None


class QAState(IntEnum):
    NONE = auto()
    UNDEFINED = auto()
    FAIL = auto()
    USABLE = auto()
    PASS = auto()


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


class ObservationClass(IntEnum):
    """
    Note that the order of these is specific and deliberate: they are listed in
    preference order for observation classes, and hence, should not be rearranged.
    """
    SCIENCE = auto()
    PROG_CAL = auto()
    PARTNER_CAL = auto()
    ACQ = auto()
    ACQ_CAL = auto()


@dataclass(frozen=True)
class InstrumentConfiguration:
    name: str
    resources: Set[Resource]


@dataclass
class Observation:
    id: str
    internal_id: str
    title: str
    site: Site
    status: ObservationStatus
    active: bool
    priority: Priority
    instrument_configuration: InstrumentConfiguration
    setuptime_type: SetupTimeType
    acq_overhead: timedelta
    exec_time: timedelta
    program_used: timedelta
    partner_used: timedelta

    # TODO: This will be handled differently between OCS and GPP.
    # TODO: 1. In OCS, when the sequence is examined, the ObservationClasses of the
    # TODO:    individual observes (sequence steps) will be analyzed and the highest
    # TODO:    precedence one will be set for the observation (based on the earliest
    # TODO:    ObservationClass in the enum).
    # TODO: 2. In GPP, this information will be handled automatically and require no
    # TODO:    special processing.
    # TODO: Should this be Optional?
    obs_class: ObservationClass

    targets: List[Target]
    guide_stars: Mapping[Resource, Target]
    sequence: List[Atom]
    constraints: Constraints
    too_type: Optional[TooType] = None

    def total_used(self) -> timedelta:
        return self.program_used + self.partner_used

    def required_resources(self) -> Set[Resource]:
        return {r for a in self.sequence for r in a.required_resources}

    def wavelengths(self) -> Set[float]:
        return {c.wavelength for c in self.sequence}

    def constraints(self) -> Set[Constraints]:
        return {self.constraints}

    @staticmethod
    def _select_obsclass(classes: List[ObservationClass]) -> Optional[ObservationClass]:
        """
        Given a list of non-empty ObservationClasses, determine which occurs with
        highest precedence in the ObservationClasses enum, i.e. has the lowest index.

        This will be used when examining the sequence for atoms.

        TODO: Move this to the ODB program extractor as the logic is used there.
        TODO: Remove from Bryan's atomizer.
        """
        return min(classes, default=None)

    @staticmethod
    def _select_qastate(qastates: List[QAState]) -> Optional[QAState]:
        """
        Given a list of non-empty QAStates, determine which occurs with
        highest precedence in the QAStates enum, i.e. has the lowest index.

        TODO: Move this to the ODB program extractor as the logic is used there.
        TODO: Remove from Bryan's atomizer.
        """
        return min(qastates, default=None)


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
    children: Union[List[Group], Observation]

    def __post_init__(self):
        super().__post_init__()
        if self.number_to_observe <= 0:
            msg = f'Group {self.group_name} specifies non-positive {self.number_to_observe} children to be observed.'
            logging.error(msg)
            raise ValueError(msg)

    def required_resources(self) -> Set[Resource]:
        return {r for c in self.children for r in c.required_resources()}

    def wavelengths(self) -> Set[float]:
        return {w for c in self.children for w in c.wavelengths()}

    def constraints(self) -> Set[Constraints]:
        return {cs for c in self.children for cs in c.constraints()}

    def __len__(self):
        return 1 if isinstance(self.children, Observation) else len(self.children)


class AndOption(Enum):
    CONSEC_ORDERED = auto()
    CONSEC_ANYORDER = auto()
    NIGHT_ORDERED = auto()
    NIGHT_ANYORDER = auto()
    ANYORDER = auto()
    CUSTOM = auto()


@dataclass
class AndGroup(NodeGroup):
    group_option: AndOption
    previous: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        if self.number_to_observe != len(self.children):
            msg = f'AND group {self.group_name} specifies {self.number_to_observe} children to be observed but has '\
                  f'{len(self.children)} children.'
            logging.error(msg)
            raise ValueError(msg)


@dataclass
class OrGroup(NodeGroup):
    def __post_init__(self):
        super().__post_init__()
        if self.number_to_observe >= len(self.children):
            msg = f'OR group {self.group_name} specifies {self.number_to_observe} children to be observed but has '\
                  f'{len(self.children)} children.'
            logging.error(msg)
            raise ValueError(msg)


class Band(IntEnum):
    BAND1 = 1
    BAND2 = 2
    BAND3 = 3
    BAND4 = 4


class ProgramMode(Enum):
    """
    Main operational mode, which is one of:
    * Queue
    * Classical
    * Priority Visitor (hybrid mode between queue and classical)
    """
    QUEUE = auto()
    CLASSICAL = auto()
    PV = auto()


@dataclass(frozen=True)
class ProgramType:
    abbreviation: str
    name: str
    isScience: bool = True


class ProgramTypes(Enum):
    C = ProgramType('C', 'Classical')
    CAL = ProgramType('CAL', 'Calibration', False)
    DD = ProgramType('DD', "Director's Time")
    DS = ProgramType('DS', 'Demo Science')
    ENG = ProgramType('ENG', 'Engineering', False)
    FT = ProgramType('FT', 'Fast Turnaround')
    LP = ProgramType('LP', 'Large Program')
    Q = ProgramType('Q', 'Queue')
    SV = ProgramType('SV', 'System Verification')


@dataclass(unsafe_hash=True)
class Program:
    """
    Representation of a program.

    The FUZZY_BOUNDARY is a constant that allows for a fuzzy boundary for a program's
    start and end times.
    """
    id: str
    internal_id: str
    band: Band
    thesis: bool
    mode: ProgramMode
    type: ProgramTypes
    start_time: datetime
    end_time: datetime
    allocated_time: Set[TimeAllocation]
    root_group: Group
    too_type: Optional[TooType] = None

    FUZZY_BOUNDARY: ClassVar[timedelta] = timedelta(days=14)

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
