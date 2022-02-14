# NOTE: In order to use numpy.typing, this file requires the 1.21 numpy package to be installed via:
# pip install numpy==1.21.4

import logging
from abc import ABC, abstractmethod
from astropy.coordinates import EarthLocation, UnknownSiteException
from astropy.time import Time, TimeDelta
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, IntEnum, auto
import numpy.typing as npt
from pytz import timezone, UnknownTimeZoneError
from typing import ClassVar, List, Mapping, Optional, Set, Union


class SiteInformation:
    def __init__(self,
                 name: str,
                 coordinate_center: str,
                 astropy_lookup: str = None):
        """
        AstroPy location lookups for Gemini North and South are of the form:
        * gemini_north
        * gemini_south
        This conversion will happen automatically if astropy_lookup is None.

        If necessary, other observatories should provide hard astropy_lookup values.

        The following is also included here:
        * name: the name of the site in human-readable format
        * timezone: time zone information
        * location: the AstroPy location lookup (astropy.coordinates.earth) of the site
        * coordinate_center: coordinate center for Ephemeris lookups
        """
        if astropy_lookup is None:
            astropy_lookup = name.lower().replace(' ', '_')

        self.name = name
        self.coordinate_center = coordinate_center

        try:
            self.location = EarthLocation.of_site(astropy_lookup)
        except UnknownSiteException:
            logging.error(f'Unknown site lookup: {astropy_lookup}')

        timezone_info = self.location.info.meta['timezone']
        try:
            self.timezone = timezone(timezone_info)
        except UnknownTimeZoneError:
            logging.error(f'Unknown time zone lookup: {timezone_info}')


class Site(Enum):
    """
    The sites belonging to the observatory using the Scheduler.

    This will have to be customized by a given observatory if used independently
    of Gemini.
    """
    GN = SiteInformation('Gemini North', '568@399')
    GS = SiteInformation('Gemini South', 'I11@399')


class SemesterHalf(Enum):
    """
    Gemini typically schedules programs for two semesters per year, namely A and B.
    For other observatories, this logic might have to be substantially changed.
    """
    A = 'A'
    B = 'B'


@dataclass
class Semester:
    """
    A semester is a period for which programs may be submitted to Gemini and consists of:
    * A four digit year
    * Two semesters during each year, indicated by the SemesterHalf
    """
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
    The time accounting codes for the possible partner submissions or internal program
    types used at Gemini, also known as categories.

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


@dataclass(unsafe_hash=True)
class TimeAllocation:
    """
    Time allocation information for a given category for a program.
    Programs may be sponsored by multiple categories with different amounts
    of time awarded. This class maintains information about the time awarded
    and the time that has been used, divided between program time and partner
    calibration time. The time used is calculated as a ratio of the awarded time
    for this category to the total time awarded to the program.
    """
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

    INFINITE_DURATION_FLAG: ClassVar[int] = -1
    INFINITE_DURATION: ClassVar[int] = timedelta.max
    FOREVER_REPEATING: ClassVar[int] = -1
    NON_REPEATING: ClassVar[int] = 0
    NO_PERIOD: ClassVar[Optional[timedelta]] = None

    # A number to be used by the Scheduler to represent infinite repeats from the
    # perspective of the OCS: if FOREVER_REPEATING is selected, then it is converted
    # into this for calculation purposes.
    OCS_INFINITE_REPEATS: ClassVar[int] = 1000


class SkyBackground(float, Enum):
    """
    Bins for observation sky background requirements or current conditions.
    """
    SB20 = 0.2
    SB50 = 0.5
    SB80 = 0.8
    SBANY = 1.0


class CloudCover(float, Enum):
    """
    Bins for observation cloud cover requirements or current conditions.
    """
    CC50 = 0.5
    CC70 = 0.7
    CC80 = 0.8
    CCANY = 1.0


class ImageQuality(float, Enum):
    """
    Bins for observation image quality requirements or current conditions.
    """
    IQ20 = 0.2
    IQ70 = 0.7
    IQ85 = 0.85
    IQANY = 1.0


class WaterVapor(float, Enum):
    """
    Bins for observation water vapor requirements or current conditions.
    """
    WV20 = 0.2
    WV50 = 0.5
    WV80 = 0.8
    WVANY = 1.0


class Strehl(float, Enum):
    """
    The Strehl ratio is a measure of the quality of optical image formation.
    Used variously in situations where optical resolution is compromised due to lens aberrations or due to imaging
    through the turbulent atmosphere, the Strehl ratio has a value between 0 and 1, with a hypothetical, perfectly
    unaberrated optical system having a Strehl ratio of 1. (Source: Wikipedia.)
    """
    S00 = 0.0
    S02 = 0.2
    S04 = 0.4
    S06 = 0.6
    S08 = 0.8
    S10 = 1.0


class ElevationType(IntEnum):
    """
    The type of elevation constraints in the observing conditions.
    """
    NONE = auto()
    HOUR_ANGLE = auto()
    AIRMASS = auto()


@dataclass(order=True)
class Conditions:
    """
    A set of conditions.

    Note that we make this dataclass eq and ordered so that we can compare one
    set of conditions with another to see if one satisfies the other.

    This should be done via:
    current_conditions <= required_conditions.
    """
    cc: CloudCover
    iq: ImageQuality
    sb: SkyBackground
    wv: WaterVapor


@dataclass
class Constraints:
    """
    The constraints required for an observation to be performed.
    """
    conditions: Conditions
    # constrast: Constrast
    elevation_type: ElevationType
    elevation_min: float
    elevation_max: float
    timing_windows: List[TimingWindow]
    # clearance_windows: Optional[List[ClearanceWindow]] = None
    strehl: Optional[Strehl] = None

    def __post_init__(self):
        """
        Convert the timing window information to more natural units, i.e. a list of
        AstroPy Time, which is for more convenient processing.

        This creates a property on the Constraints called ot_timing_windows.

        TODO: Do we need this?
        """
        self.ot_timing_windows: List[Time] = []

        # # Collect the timing window information as arrays from the TimingWindow list.
        # starts = (tw.start for tw in self.timing_windows)
        # durations = (tw.duration for tw in self.timing_windows)
        # repeats = (tw.repeat for tw in self.timing_windows)
        # periods = (tw.period for tw in self.timing_windows)
        #
        # for (s, d, r, p) in zip(starts, durations, repeats, periods):
        #     start = Time(s)
        #     duration = TimeDelta.max if d == -1 else TimeDelta(d)
        #     repeat = TimingWindow.OCS_INFINITE_REPEATS if r == TimingWindow.FOREVER_REPEATING else max(1, r)
        #     period = None if p is None else TimeDelta(p)
        #
        #     for i in range(repeat):
        #         window_start = start if period is None else start + i * period
        #         window_end = window_start + duration
        #
        #         # TODO: This does not seem correct.
        #         # TODO: We should be inserting TimingWindow into this list, and not these
        #         # TODO: AstroPy Time objects, which are unexpected and cannot be indexed.
        #         self.ot_timing_windows.append(Time([window_start, window_end]))


class MagnitudeSystem(Enum):
    """
    List of magnitude systems associated with magnitude bands.
    """
    VEGA = auto()
    AB = auto()
    JY = auto()


@dataclass(frozen=True)
class MagnitudeBand:
    """
    THIS CLASS SHOULD NOT BE INSTANTIATED.
    They are fully enumerated in MagnitudeBands, so they should be looked up by name there.

    Values for center and width are specified in microns.
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
    """
    A magnitude value in a particular band.
    """
    band: MagnitudeBands
    value: float
    error: Optional[float] = None


class TargetType(Enum):
    """
    The type associated with a target in an observation.
    """
    BASE = auto()
    USER = auto()
    BLIND_OFFSET = auto()
    OFF_AXIS = auto()
    TUNING_STAR = auto()
    GUIDESTAR = auto()
    OTHER = auto()


class GuideSpeed(IntEnum):
    """
    How quickly a guider can guide on a guide star.
    """
    SLOW = auto()
    MEDIUM = auto()
    FAST = auto()


class TargetTag(Enum):
    """
    A tag used by nonsidereal targets to indicate their type.
    """
    COMET = auto()
    ASTEROID = auto()
    MAJOR_BODY = auto()


# Type alias for a target name.
TargetName = str


@dataclass
class Target(ABC):
    """
    Basic target information.
    """
    name: TargetName
    magnitudes: Set[Magnitude]
    type: TargetType

    def __post_init__(self):
        pass

    def guide_speed(self) -> GuideSpeed:
        """
        Calculate the guide speed for this target.
        """
        ...


@dataclass
class SiderealTarget(Target):
    """
    For a SiderealTarget, we have an RA and Dec and then proper motion information
    to calculate the exact position.

    RA and Dec should be specified in decimal degrees.
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
    For a NonsiderealTarget, we have a HORIZONS designation to indicate the lookup
    information, a tag to determine the type of target, and arrays of ephemerides
    to specify the position.
    """
    des: str
    tag: TargetTag
    ra: npt.NDArray[float]
    dec: npt.NDArray[float]

    def __post_init__(self):
        super().__post_init__()


@dataclass(unsafe_hash=True, frozen=True)
class Resource:
    """
    This is a general observatory resource.
    It can consist of a guider, an instrument, or a part of an instrument,
    or even a personnel and is used to determine what observations can be
    performed at a given time based on the resource availability.
    """
    id: str
    description: Optional[str] = None

    def __eq__(self, other):
        return isinstance(other, Resource) and self.id == other.id


class QAState(IntEnum):
    """
    These correspond to the QA States in the OCS for Observations.
    Entries in the obs log should be made uppercase for lookups into
    this enum.
    """
    NONE = auto()
    UNDEFINED = auto()
    FAIL = auto()
    USABLE = auto()
    PASS = auto()


@dataclass
class InstConfig:
    """
    Atom instrument configuration.
    All of these are Resource objects, but divided into categories for convenience.
    Wavelengths are the exception, and must be specified in microns.
    TODO: Is this necessary, or can we just have a Set[Resource] in Atom?
    """
    inst: Resource
    fpu: Set[Resource]
    disperser: Set[Resource]
    filter: Set[Resource]
    wavelength: Set[float]


@dataclass
class Atom:
    """
    Atom information, where an atom is the smallest schedulable set of steps
    such that useful science can be obtained from performing them.
    Wavelengths must be specified in microns.
    """
    id: int
    exec_time: timedelta
    prog_time: timedelta
    part_time: timedelta
    observed: bool
    qa_state: QAState
    guide_state: bool

    # TODO: Select between resources / wavelength or inst_config model.
    resources: Set[Resource]
    wavelength: Set[float]
    # inst_config: InstConfig


class ObservationStatus(IntEnum):
    """
    The status of an observation as indicated in the Observing Tool / ODB.
    """
    NEW = auto()
    INCLUDED = auto()
    PROPOSED = auto()
    APPROVED = auto()
    FOR_REVIEW = auto()
    READY = auto()
    ONGOING = auto()
    OBSERVED = auto()
    INACTIVE = auto()
    PHASE2 = auto()


class Priority(IntEnum):
    """
    An observation's priority.
    Note that these are ordered specifically so that we can compare them.
    """
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()


class TooType(IntEnum):
    """
    The target-of-opportunity type for a program and for an observation.
    These are ordered specifically so that we can compare them.

    The INTERRUPT is considered the highest level of TooType, followed by RAPID, and then STANDARD.

    Thus, a Program with a RAPID type, for example, can contain RAPID and STANDARD Observations,
    but not INTERRUPT ones.

    The values and ordering on them should NOT be changed as this will break functionality.
    """
    STANDARD = auto()
    RAPID = auto()
    INTERRUPT = auto()


class SetupTimeType(IntEnum):
    """
    The setup time type for an observation.
    """
    FULL = auto()
    REACQUISITION = auto()
    NONE = auto()


class ObservationClass(IntEnum):
    """
    The class of an observation.

    Note that the order of these is specific and deliberate: they are listed in
    preference order for observation classes, and hence, should not be rearranged.
    These correspond to the values in the OCS when made uppercase.
    """
    SCIENCE = auto()
    PROGCAL = auto()
    PARTNERCAL = auto()
    ACQ = auto()
    ACQCAL = auto()
    DAYCAL = auto()


# Alias for observation identifier.
ObservationID = str


@dataclass
class Observation:
    """
    Representation of an observation.
    Non-obvious fields are documented below.
    * id should represent the observation's ID, e.g. GN-2018B-Q-101-123.
    * internal_id is the key associated with the observation
    * order refers to the order of the observation in either its group or the program
    * targets should contain a complete list of all targets associated with the observation,
      with the base being in the first position
    * guiding is a map between guide probe resources and their targets
    """
    id: ObservationID
    internal_id: str
    order: int
    title: str
    site: Site
    status: ObservationStatus
    active: bool
    priority: Priority
    resources: Set[Resource]
    setuptime_type: SetupTimeType
    acq_overhead: timedelta
    exec_time: timedelta

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
    guiding: Mapping[Resource, Target]
    sequence: List[Atom]

    # Some observations do not have constraints, e.g. GN-208A-FT-103-6.
    constraints: Optional[Constraints]

    too_type: Optional[TooType] = None

    def total_used(self) -> timedelta:
        """
        Total program time used: includes program time and partner time.
        """
        return self.program_used() + self.partner_used()

    def required_resources(self) -> Set[Resource]:
        """
        The required resources for an observation based on the sequence's needs.
        """
        return self.resources | {r for a in self.sequence for r in a.resources}

    def wavelengths(self) -> Set[float]:
        """
        The set of wavelengths included in the sequence.
        Wavelengths are specified in microns.
        """
        return {w for c in self.sequence for w in c.wavelength}

    def constraints(self) -> Set[Constraints]:
        """
        A set of the constraints required by the observation.
        In the case of an observation, this is just the (optional) constraints.
        """
        return {self.constraints} if self.constraints is not None else {}

    def program_used(self) -> timedelta:
        """
        We roll this information up from the atoms as it will be calculated
        during the GreedyMax algorithm. Note that it is also available directly
        from the OCS, which is used to populate the time allocation.
        """
        return sum(atom.prog_time for atom in self.sequence)

    def partner_used(self) -> timedelta:
        """
        We roll this information up from the atoms as it will be calculated
        during the GreedyMax algorithm. Note that it is also available directly
        from the OCS, which is used to populate the time allocation.
        """
        return sum(atom.part_time for atom in self.sequence)

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

    def __len__(self):
        """
        This is to treat observations the same as groups and is a bit of a hack.
        Observations are to be placed in AND Groups of size 1 for scheduling purposes.
        """
        return 1


# Since Python doesn't allow classes to self-reference, we have to make a basic group
# from which to subclass.
@dataclass
class Group(ABC):
    """
    This is the base implementation of AND / OR Groups.
    Python does not allow classes to self-reference unless in static contexts,
    so we make a very simple base class to self-reference from subclasses since
    we need this functionality to allow for group nesting.

    * id: the identification of the group
    * group_name: a human-readable name of the group
    * number_to_observe: the number of children in the group that must be observed for the
      group to be considered complete
    * delay_min: used in cadences
    * delay_max: used in cadences
    """
    id: str
    group_name: str
    number_to_observe: int
    delay_min: timedelta
    delay_max: timedelta

    def __post_init__(self):
        pass

    @abstractmethod
    def required_resources(self) -> Set[Resource]:
        """
        This method should be implemented to return all the Resources required by
        this group and its descendents.
        """
        ...

    @abstractmethod
    def wavelengths(self) -> Set[float]:
        """
        This method should be implemented to return all the wavelengths used by
        this group and its descendents.
        """
        ...

    @abstractmethod
    def constraints(self) -> Set[Constraints]:
        """
        This method should be used to return all the sets of Conditions required by
        this group and its descendents.
        """
        ...

    @abstractmethod
    def observations(self) -> List[Observation]:
        """
        This method should be used to return all the sets of Observations contained
        in this group and its descendents.
        """
        ...


@dataclass
class NodeGroup(Group, ABC):
    """
    A NodeGroup is the fundamental implementation of a group, i.e. a group that
    contains children, which can either be:
    1. A single observation (in which case, the group should be an AND group); or
    2. A list of other groups (in which case, the group can be either an AND or OR group).
    Note that it is still abstract and cannot be instantiated.

    The distinction between Group and NodeGroup is made so that NodeGroup can
    reference Group in its members, since Python classes cannot be self-referential.
    """
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

    def observations(self) -> List[Observation]:
        return [self.children] if isinstance(self.children, Observation) else []

    def __len__(self):
        return 1 if isinstance(self.children, Observation) else len(self.children)


class AndOption(Enum):
    """
    Different options available for ordering AND group children.
    CUSTOM is used for cadences.
    """
    CONSEC_ORDERED = auto()
    CONSEC_ANYORDER = auto()
    NIGHT_ORDERED = auto()
    NIGHT_ANYORDER = auto()
    ANYORDER = auto()
    CUSTOM = auto()


@dataclass
class AndGroup(NodeGroup):
    """
    The concrete implementation of an AND group.
    It requires an AndOption to specify how its observations should be handled,
    and a previous (which should be an index into the group's children to indicate
    the previously observed child, or None if none of the children have yet been
    observed).
    """
    group_option: AndOption
    previous: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        if self.number_to_observe != len(self.children):
            msg = f'AND group {self.group_name} specifies {self.number_to_observe} children to be observed but has '\
                  f'{len(self.children)} children.'
            logging.error(msg)
            raise ValueError(msg)
        if self.previous is not None and (self.previous < 0 or self.previous >= len(self.children)):
            msg = f'AND group {self.group_name} has {len(self.children)} children and an illegal previous value of '\
                  f'{self.previous}'
            logging.error(msg)
            raise ValueError(msg)


@dataclass
class OrGroup(NodeGroup):
    """
    The concrete implementation of an OR group.
    The restrictions on an OR group is that it must explicitly require not all
    of its children to be observed.
    """
    def __post_init__(self):
        super().__post_init__()
        if self.number_to_observe >= len(self.children):
            msg = f'OR group {self.group_name} specifies {self.number_to_observe} children to be observed but has '\
                  f'{len(self.children)} children.'
            logging.error(msg)
            raise ValueError(msg)


class Band(IntEnum):
    """
    Program band.
    """
    BAND1 = 1
    BAND2 = 2
    BAND3 = 3
    BAND4 = 4


class ProgramMode(IntEnum):
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
    """
    Represents the information encompassing the type of program.
    * abbreviation: the code used by the program type (e.g. Q, C, FT, LP)
    * name: user readable representation of the program type
    * is_science: indicates if this program type is a science program

    NOTE that ProgramType instances should NEVER be explicitly created.
    All of the valid ProgramType instances are contained in the ProgramTypes enum
    and should be accessed from there.
    """
    abbreviation: str
    name: str
    is_science: bool = True


class ProgramTypes(Enum):
    """
    A complete list of the ProgramType instances used by Gemini.
    As mentioned in ProgramType, ProgramType should never be instantiated
    outside of this enum: instead, ProgramType instances should be retrieved
    from here.
    """
    C = ProgramType('C', 'Classical')
    CAL = ProgramType('CAL', 'Calibration', False)
    DD = ProgramType('DD', "Director's Time")
    DS = ProgramType('DS', 'Demo Science')
    ENG = ProgramType('ENG', 'Engineering', False)
    FT = ProgramType('FT', 'Fast Turnaround')
    LP = ProgramType('LP', 'Large Program')
    Q = ProgramType('Q', 'Queue')
    SV = ProgramType('SV', 'System Verification')


# Type alias for program ID.
ProgramID = str


@dataclass(unsafe_hash=True)
class Program:
    """
    Representation of a program.

    The FUZZY_BOUNDARY is a constant that allows for a fuzzy boundary for a program's
    start and end times.
    """
    id: ProgramID
    internal_id: str
    # Some programs do not have a typical name and thus cannot be associated with a semester.
    semester: Optional[Semester]
    band: Band
    thesis: bool
    mode: ProgramMode
    type: Optional[ProgramTypes]
    start: datetime
    end: datetime
    allocated_time: Set[TimeAllocation]
    root_group: AndGroup
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

    def observations(self) -> List[Observation]:
        return self.root_group.observations()


@dataclass(frozen=True)
class Visit:
    """
    A visit is a scheduled piece of an observation.
    It can be no less than a single atom.

    TODO: This will probably require some refinement as we progress.
    """
    start_time: datetime
    end_time: datetime
    observation_id: str
    first_atom_id: str
    last_atom_id: str
    comment: str
    setuptime_type: SetupTimeType


@dataclass
class Plan:
    """
    A complete plan for each site, mapping an observation period to the list
    of visits to be performed during that period.

    TODO: This will probably require some refinement as we progress.
    """
    scheduled_atoms: Mapping[Site, Mapping[ObservingPeriod, List[Visit]]]
