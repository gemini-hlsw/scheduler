from .atom import *
from .constraints import *
from .group import *
from .magnitude import *
from .observation import *
from .program import *
from .qastate import *
from .resource import *
from .semester import *
from .site import *
from .target import *
from .timeallocation import *
from .timingwindow import *
from .too import *

# Type alias for night indices.
NightIndex = int


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
        except UnknownSiteException as e:
            msg = f'Unknown site lookup: {astropy_lookup}.'
            raise ValueError(e, msg)

        timezone_info = self.location.info.meta['timezone']
        try:
            self.timezone = timezone(timezone_info)
        except UnknownTimeZoneError as e:
            msg = f'Unknown time zone lookup: {timezone_info}.'
            raise ValueError(e, msg)


class Site(Enum):
    """
    The sites belonging to the observatory using the Scheduler.

    This will have to be customized by a given observatory if used independently
    of Gemini.
    """
    GN = SiteInformation('Gemini North', '568@399')
    GS = SiteInformation('Gemini South', 'I11@399')


ALL_SITES = frozenset(s for s in Site)


class SemesterHalf(Enum):
    """
    Gemini typically schedules programs for two semesters per year, namely A and B.
    For other observatories, this logic might have to be substantially changed.
    """
    A = 'A'
    B = 'B'


@dataclass(unsafe_hash=True)
class Semester:
    """
    A semester is a period for which programs may be submitted to Gemini and consists of:
    * A four digit year
    * Two semesters during each year, indicated by the SemesterHalf
    """
    year: int
    half: SemesterHalf

    def __str__(self):
        return f'{self.year}{self.half.value}'


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

    # For infinite duration, use the length of an LP.
    INFINITE_DURATION_FLAG: ClassVar[int] = -1
    INFINITE_DURATION: ClassVar[int] = timedelta(days=3 * 365, hours=24)
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


@dataclass(order=True, frozen=True)
class Conditions:
    """
    A set of conditions.

    Note that we make this dataclass eq and ordered so that we can compare one
    set of conditions with another to see if one satisfies the other.

    This should be done via:
    current_conditions <= required_conditions.
    """
    cc: Union[npt.NDArray[CloudCover], CloudCover]
    iq: Union[npt.NDArray[ImageQuality], ImageQuality]
    sb: Union[npt.NDArray[SkyBackground], SkyBackground]
    wv: Union[npt.NDArray[WaterVapor], WaterVapor]

    # Least restrictive conditions.
    @classmethod
    def least_restrictive(cls) -> 'Conditions':
        """
        Return the least possible restrictive conditions.
        """
        return cls(cc=CloudCover.CCANY,
                   iq=ImageQuality.IQANY,
                   sb=SkyBackground.SBANY,
                   wv=WaterVapor.WVANY)

    def __post_init__(self):
        """
        Ensure that if any arrays are specified, all values are specified arrays of the same size.
        """
        is_uniform = len({np.isscalar(self.cc), np.isscalar(self.iq), np.isscalar(self.sb), np.isscalar(self.wv)}) == 1
        if not is_uniform:
            raise ValueError(f'Conditions have a mixture of array and scalar types: {self}')

        are_arrays = isinstance(self.cc, np.ndarray)
        if are_arrays:
            uniform_lengths = len({self.cc.size, self.iq.size, self.sb.size, self.wv.size}) == 1
            if not uniform_lengths:
                raise ValueError(f'Conditions have a variable number of array sizes: {self}')

    @staticmethod
    def most_restrictive_conditions(conditions: Sequence['Conditions']) -> 'Conditions':
        """
        Given an iterable of conditions, find the most restrictive amongst the set.
        If no conditions are given, return the most flexible conditions possible.
        """
        if len(conditions) == 0:
            return Conditions.least_restrictive()
        min_cc = min(flatten(c.cc for c in conditions), default=CloudCover.CCANY)
        min_iq = min(flatten(c.iq for c in conditions), default=ImageQuality.IQANY)
        min_sb = min(flatten(c.sb for c in conditions), default=SkyBackground.SBANY)
        min_wv = min(flatten(c.wv for c in conditions), default=WaterVapor.WVANY)
        return Conditions(cc=min_cc, iq=min_iq, sb=min_sb, wv=min_wv)

    def __len__(self):
        """
        For array values, return the length of the arrays.
        For scalar values, return a length of 1.
        """
        return len(self.cc) if isinstance(self.cc, np.ndarray) else 1


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

    # Default airmass values to use for elevation constraints if:
    # 1. The Constraints are not present in the Observation at all; or
    # 2. The elevation_type is set to NONE.
    DEFAULT_AIRMASS_ELEVATION_MIN: ClassVar[float] = 1.0
    DEFAULT_AIRMASS_ELEVATION_MAX: ClassVar[float] = 2.3


@dataclass(order=True, eq=True, frozen=True)
class Variant:
    """
    A weather variant.
    wind_speed should be in m / s.
    TODO: No idea what time blocks are. Note this could be a list or a single value.
    TODO: Because of this, we cannot hash Variants., which is problematic.
    """
    iq: Union[npt.NDArray[ImageQuality], ImageQuality]
    cc: Union[npt.NDArray[CloudCover], CloudCover]
    wv: Union[npt.NDArray[WaterVapor], WaterVapor]
    wind_dir: Angle
    wind_sep: Angle
    wind_spd: Quantity

    # time_blocks: Time

    def __post_init__(self):
        """
        Ensure that if any arrays are specified, all values are specified arrays of the same size.
        """
        is_uniform = len({np.isscalar(self.cc), np.isscalar(self.iq), np.isscalar(self.wv)}) == 1
        if not is_uniform:
            raise ValueError(f'Variant has a mixture of array and scalar types: {self}')

        are_arrays = isinstance(self.cc, np.ndarray)
        array_lengths = {np.asarray(self.wind_dir).size,
                         np.asarray(self.wind_sep).size,
                         np.asarray(self.wind_spd).size}
        if are_arrays:
            uniform_lengths = len({len(self.cc), len(self.iq), len(self.wv)}.union(array_lengths)) == 1
        else:
            uniform_lengths = len(array_lengths) == 1
        if not uniform_lengths:
            raise ValueError(f'Variants has a variable number of array sizes: {self}')


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

    NOTE: The proper motion adjusted coordinates can be found in the TargetInfo in coord.
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
    # TODO: Not in original mini-model description, but returned by OCS.
    CHECK = auto()


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
    resources: Set[Resource]
    wavelengths: Set[float]


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


class ObservationStatus(IntEnum):
    """
    The status of an observation as indicated in the Observing Tool / ODB.
    """
    NEW = auto()
    INCLUDED = auto()
    PROPOSED = auto()
    APPROVED = auto()
    FOR_REVIEW = auto()
    # TODO: Not in original mini-model description, but returned by OCS.
    ON_HOLD = auto()
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
    NONE = auto()
    REACQUISITION = auto()
    FULL = auto()


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


# Type alias for program ID.
ProgramID = str

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
    setuptime_type: SetupTimeType
    acq_overhead: timedelta

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

    def base_target(self) -> Optional[Target]:
        """
        Get the base target for this Observation if it has one, and None otherwise.
        """
        return next(filter(lambda t: t.type == TargetType.BASE, self.targets), None)

    def exec_time(self) -> timedelta:
        """
        Total execution time for the program, which is sum across atoms and the acquisition overhead.
        """
        return sum((atom.exec_time for atom in self.sequence), timedelta()) + self.acq_overhead

    def total_used(self) -> timedelta:
        """
        Total program time used: includes program time and partner time.
        """
        return self.program_used() + self.partner_used()

    def required_resources(self) -> Set[Resource]:
        """
        The required resources for an observation based on the sequence's needs.
        """
        return self.guiding.keys() | {r for a in self.sequence for r in a.resources}

    def wavelengths(self) -> Set[float]:
        """
        The set of wavelengths included in the sequence.
        """
        return {w for c in self.sequence for w in c.wavelengths}

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
        return sum((atom.prog_time for atom in self.sequence), start=timedelta())

    def partner_used(self) -> timedelta:
        """
        We roll this information up from the atoms as it will be calculated
        during the GreedyMax algorithm. Note that it is also available directly
        from the OCS, which is used to populate the time allocation.
        """
        return sum((atom.part_time for atom in self.sequence), start=timedelta())

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

    def __eq__(self, other: 'Observation') -> bool:
        """
        We override the equality checker created by @dataclass to temporarily skip sequence
        comparison in test cases until the atom creation process is finish.
        """

        return (dict((k, v) for k, v in self.__dict__.items() if k != 'sequence') ==
                dict((k, v) for k, v in other.__dict__.items() if k != 'sequence'))


# Type alias for group ID.
GroupID = str


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
    id: GroupID
    group_name: str
    number_to_observe: int
    delay_min: timedelta
    delay_max: timedelta
    children: Union[List['Group'], Observation]

    def __post_init__(self):
        if self.number_to_observe <= 0:
            msg = f'Group {self.group_name} specifies non-positive {self.number_to_observe} children to be observed.'
            raise ValueError(msg)

    def subgroup_ids(self) -> Set[GroupID]:
        if isinstance(self.children, Observation):
            return set()
        else:
            return {subgroup.id for subgroup in self.children}

    def sites(self) -> Set[Site]:
        if isinstance(self.children, Observation):
            return {self.children.site}
        else:
            return set.union(*[s.sites() for s in self.children])

    def required_resources(self) -> Set[Resource]:
        return {r for c in self.children for r in c.required_resources()}

    def wavelengths(self) -> Set[float]:
        return {w for c in self.children for w in c.wavelengths()}

    def constraints(self) -> Set[Constraints]:
        return {cs for c in self.children for cs in c.constraints()}

    def observations(self) -> List[Observation]:
        if isinstance(self.children, Observation):
            return [self.children]
        else:
            return [o for g in self.children for o in g.observations()]
    
    def total_used(self) -> timedelta:
        return sum([o.total_used() for o in self.observations()], start=timedelta())

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
class AndGroup(Group):
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
            msg = f'AND group {self.group_name} specifies {self.number_to_observe} children to be observed but has ' \
                  f'{len(self.children)} children.'
            raise ValueError(msg)
        if self.previous is not None and (self.previous < 0 or self.previous >= len(self.children)):
            msg = f'AND group {self.group_name} has {len(self.children)} children and an illegal previous value of ' \
                  f'{self.previous}'
            raise ValueError(msg)


@dataclass
class OrGroup(Group):
    """
    The concrete implementation of an OR group.
    The restrictions on an OR group is that it must explicitly require not all
    of its children to be observed.
    """

    def __post_init__(self):
        super().__post_init__()
        if self.number_to_observe >= len(self.children):
            msg = f'OR group {self.group_name} specifies {self.number_to_observe} children to be observed but has ' \
                  f'{len(self.children)} children.'
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
        return sum((t.program_awarded for t in self.allocated_time), timedelta())

    def program_used(self) -> timedelta:
        return sum((t.program_used for t in self.allocated_time), timedelta())

    def partner_awarded(self) -> timedelta:
        return sum((t.partner_awarded for t in self.allocated_time), timedelta())

    def partner_used(self) -> timedelta:
        return sum((t.partner_used for t in self.allocated_time), timedelta())

    def total_awarded(self) -> timedelta:
        return sum((t.total_awarded() for t in self.allocated_time), timedelta())

    def total_used(self) -> timedelta:
        return sum((t.total_used() for t in self.allocated_time), timedelta())

    def observations(self) -> List[Observation]:
        return self.root_group.observations()


class Plans:
    """
    A collection of Nightly Plan from a corresponding Site
    """
    def __init__(self, night_events):
        self.durations = [len(night_events.times[idx]) for idx, _ in enumerate(night_events.time_grid)]
        self.start_times = [night_events.local_times[idx][0] for idx, _ in enumerate(night_events.time_grid)]
        self.end_times = [night_events.local_times[idx][-1] for idx, _ in enumerate(night_events.time_grid)]
        self.time_slot_length = night_events.time_slot_length

    def __iter__(self):
        self.plans = []
        return self

    def __next__(self):
        if len(self.plans) <= len(self.durations):
            self.plans.append(Plan(next(iter(self.start_times)),
                                   next(iter(self.end_times)),
                                   slot_length=self.time_slot_length,
                                   night_duration=next(iter(self.durations))))
            return next(iter(self.plans))
        raise StopIteration


class Plan:
    """
    A 'plan' is a collection of nighly plans
    """
    def __init__(self, start: datetime, end:datetime, slot_length=1, night_duration=10):
        self.start = start
        self.end = end
        self._time_slot_length = slot_length
        self._time_slots_left = night_duration
        self._visits = []
    
    def _time2slots(self, time: datetime) -> int:
        return ceil((time.total_seconds() / 60) / self._time_slot_length.value)

    def add_group(self, group: Tuple[Group, GroupInfo]) -> NoReturn:
        self._visits.append(group)
        self._time_slots_left -= self._time2slots(group[0].total_used())
    
    def is_full(self) -> bool:
        return self._time_slots_left <= 0

    def time_slots_left(self) -> int:
        return self._time_slots_left


class Plans:
    """
    A collection of Nightly Plan from a corresponding Site
    """
    def __init__(self, night_events):
        # TODO: adding NightEvents creates a circular dependency!
        durations = list(map(len, night_events.times))
        start_times = [local_times[0] for local_times in night_events.local_times]
        end_times = [local_times[-1] for local_times in night_events.local_times]
        self.time_slot_length = night_events.time_slot_length
        self.plans = []
        for duration, start, end in zip(durations, start_times, end_times):
            self.plans.append(Plan(start, end, slot_length=self.time_slot_length,
                                   night_duration=duration))

    def __iter__(self):
        self._counter = 0
        return self

    def __next__(self):
        if self._counter <= len(self.plans):
            self._counter += 1
            return next(iter(self.plans))
        raise StopIteration


