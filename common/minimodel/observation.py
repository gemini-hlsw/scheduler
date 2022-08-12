from dataclasses import dataclass
from datetime import timedelta
from enum import auto, IntEnum
from typing import List, Mapping, Optional, Set

from .atom import Atom
from .constraints import Constraints
from .qastate import QAState
from .resource import Resource
from .site import Site
from .target import Target, TargetType
from .too import TooType

ObservationID = str


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
        Total execution time for the program, which is the sum across atoms and the acquisition overhead.
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

    def instrument(self) -> Optional[Resource]:
        """
        Returns a resource that is an instrument, if one exists.
        There should be only one.
        """
        return next(filter(lambda r: api.observatory.abstract.ObservatoryProperties.is_instrument(r),
                           self.required_resources()), None)

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
