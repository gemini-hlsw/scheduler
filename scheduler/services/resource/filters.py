# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from abc import ABC
from dataclasses import dataclass
from typing import Callable, FrozenSet, Optional, final

from lucupy.minimodel import Group, Program, ProgramID, Resource, TimeAccountingCode


# A filter applied to the programs for the night.
ProgramFilter = Callable[[Program], bool]

# A filter applied to groups.
GroupFilter = Callable[[Group], bool]


@dataclass(frozen=True)
class AbstractFilter(ABC):
    """
    Abstract superclass for all filters.
    """
    @property
    def program_filter(self) -> Optional[ProgramFilter]:
        """
        Determine the programs that can be added to the plan.
        """
        return None

    @property
    def program_priority_filter(self) -> Optional[ProgramFilter]:
        """
        Determine the programs whose scheduling groups should receive the highest priority for the plan.
        """
        return None

    @property
    def group_filter(self) -> Optional[GroupFilter]:
        """
        Determine the groups that can be added to the plan.
        """
        return None

    @property
    def group_priority_filter(self) -> Optional[GroupFilter]:
        """
        Determine the groups that should receive the highest priority for the plan.
        """
        return None


@final
@dataclass(frozen=True)
class ResourcesAvailableFilter(AbstractFilter):
    """
    Filter level: Group.

    Cases:
    * Positive (primary): Used to filter in scheduling groups whose resource needs are met.
    * Negative: Can be used if there is, e.g., a fault and a Resource is no longer available.
    """
    resources: FrozenSet[Resource]

    @property
    def group_filter(self) -> Optional[GroupFilter]:
        return lambda g: self.resources.issuperset(g.required_resources())


@final
@dataclass(frozen=True)
class TimeAccountingCodeFilter(AbstractFilter):
    """
    Filter level: Program

    Cases:
    * Positive (primary): Filter in programs who contain the one of the time accounting codes.
                          Represents, for example, South Korea (ROK) blocks.
    * Negative: No current use cases.
    """
    codes: FrozenSet[TimeAccountingCode]

    @property
    def program_filter(self) -> Optional[ProgramFilter]:
        return lambda p: any(t.category in self.codes for t in p.allocated_time)


@final
@dataclass(frozen=True)
class ProgramPermissionFilter(AbstractFilter):
    """
    Filter level: Program

    Cases:
    * Positive: The program ID must be in the list to be filtered in.
    * Negative (primary): The program ID must not be in the list to be filtered in.
    """
    program_ids: FrozenSet[ProgramID]

    @property
    def program_filter(self) -> Optional[ProgramFilter]:
        return lambda p: p.id in self.program_ids


@final
@dataclass(frozen=True)
class ProgramPriorityFilter(AbstractFilter):
    """
    Filter level: Program Priority

    Cases:
    * Positive (primary): Priority is assigned to the programs with the given IDs.
    * Negative: No current use cases.
    """
    program_ids: FrozenSet[ProgramID]

    @property
    def program_priority_filter(self) -> Optional[ProgramFilter]:
        return lambda p: p.id in self.program_ids


@final
@dataclass(frozen=True)
class ResourcePriorityFilter(AbstractFilter):
    """
    Filter level: Group Priority

    Cases:
    * Positive (primary): Priority is assigned to the groups that contain one of the listed resources.
    * Negative: No current use cases.
    """
    resources: FrozenSet[Resource]

    @property
    def group_priority_filter(self) -> Optional[GroupFilter]:
        return lambda g: any(r in self.resources for r in g.required_resources())


@final
@dataclass(frozen=True)
class NothingFilter(AbstractFilter):
    """
    Filter level: All.

    Cases:
    * Positive (primary): Everything is rejected.
    * Negative: Serves no purpose.
    """
    @property
    def program_filter(self) -> Optional[ProgramFilter]:
        return lambda _: False

    @property
    def program_priority_filter(self) -> Optional[ProgramFilter]:
        return lambda _: False

    @property
    def group_filter(self) -> Optional[GroupFilter]:
        return lambda _: False

    @property
    def group_priority_filter(self) -> Optional[GroupFilter]:
        return lambda _: False


@final
@dataclass(frozen=True)
class TooFilter(AbstractFilter):
    """
    Filter level: Program

    Cases:
    * Positive: Filter in programs that do not have a ToO type of None.
    * Negative (primary): Only allow programs that are not ToO programs. Used in nights where ToOs are not allowed.
    """
    @property
    def program_filter(self) -> Optional[ProgramFilter]:
        return lambda p: p.too_type is not None


@final
@dataclass(frozen=True)
class LgsFilter(AbstractFilter):
    """
    Filter level: TODO

    Cases:
    * Positive: TODO
    * Negative (primary): Filter out programs that require LGS.
    TODO: Unsure of how to implement this.
    """
    pass


@final
@dataclass(frozen=True)
class CompositeFilter(AbstractFilter):
    """
    Filter level: All

    Case:
    * Special: Contains both positive and negative filters, and returns the result of executing
        all filters on program or groups, both with respect to inclusion and priority.
    """
    positive_filters: FrozenSet[AbstractFilter] = frozenset()
    negative_filters: FrozenSet[AbstractFilter] = frozenset()

    @property
    def program_filter(self) -> Optional[ProgramFilter]:
        return (lambda p: all(pf.program_filter(p) for pf in self.positive_filters if pf.program_filter is not None) and
                not any(nf.program_filter(p) for nf in self.negative_filters if nf.program_filter is not None))

    @property
    def program_priority_filter(self) -> Optional[ProgramFilter]:
        return (lambda p: all(pf.program_priority_filter(p) for pf in self.positive_filters
                              if pf.program_priority_filter is not None) and
                not any(nf.program_priority_filter(p) for nf in self.negative_filters
                        if nf.program_priority_filter is not None))

    @property
    def group_filter(self) -> Optional[GroupFilter]:
        return (lambda g: all(pf.group_filter(g) for pf in self.positive_filters if pf.group_filter is not None) and
                not any(nf.group_filter(g) for nf in self.negative_filters if nf.group_filter is not None))

    @property
    def group_priority_filter(self) -> Optional[GroupFilter]:
        return (lambda g: all(pf.group_priority_filter(g) for pf in self.positive_filters
                              if pf.program_priority_filter is not None) and
                not any(nf.group_priority_filter(g) for nf in self.negative_filters
                        if nf.program_priority_filter is not None))
