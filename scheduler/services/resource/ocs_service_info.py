# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from abc import ABC
from dataclasses import dataclass
from typing import Callable, FrozenSet, Optional, final, Any

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
class ResourceFilter(AbstractFilter):
    """
    A filter that determines which scheduling groups can be run based on the resources available.
    """
    resources: FrozenSet[Resource]

    @property
    def group_filter(self) -> Optional[GroupFilter]:
        return lambda g: self.resources.issuperset(g.required_resources())


@final
@dataclass(frozen=True)
class TimeAccountingCodeFilter:
    """
    A program-level filter.

    A block associated with a partner / time accounting code.
    Only programs that have one of the specified time accounting codes are permitted.
    """
    codes: FrozenSet[TimeAccountingCode]

    @property
    def program_filter(self) -> Optional[ProgramFilter]:
        return lambda p: any(t.category in self.codes for t in p.allocated_time)


@final
@dataclass(frozen=True)
class ProgramPermissionFilter:
    """
    A program-level filter.

    Contains a list of ProgramIDs that are permitted.
    """
    program_ids: FrozenSet[ProgramID]

    @property
    def program_filter(self) -> Optional[ProgramFilter]:
        return lambda p: p in self.program_ids


@final
@dataclass(frozen=True)
class ProgramPriorityFilter:
    """
    A program-level priority filter.

    Contains a list of ProgramIDs that should be given priority.
    """
    program_ids: FrozenSet[ProgramID]

    @property
    def program_priority_filter(self) -> Optional[ProgramFilter]:
        return lambda p: p.id in self.program_ids


@final
@dataclass(frozen=True)
class ResourcePriorityFilter:
    """
    A group-level filter.

    Contains a set of Resource such that the set of scheduling groups requires one of those resources
    should be considered a higher priority than any other scheduling group.
    """
    resources: FrozenSet[Resource]

    @property
    def group_priority_filter(self) -> Optional[GroupFilter]:
        return lambda g: any(r in self.resources for r in g.required_resources())


@final
@dataclass(frozen=True)
class NothingFilter:
    """
    A filter that rejects all programs and groups.
    This can be used, for example, in shutdowns.
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
class CompositeFilter:
    """
    Contains composite filters, both positive and negative, and returns the result of executing
    all filters on a program or group, both with respect to inclusion and priority.
    """
    positive_filters: FrozenSet[AbstractFilter]
    negative_filters: FrozenSet[AbstractFilter]

    @property
    def program_filter(self) -> Optional[ProgramFilter]:
        return (lambda p: all(pf.program_filter(p) for pf in self.positive_filters) and
                not any(nf.program_filter(p) for nf in self.negative_filters))

    @property
    def program_priority_filter(self) -> Optional[ProgramFilter]:
        return (lambda p: all(pf.program_priority_filter(p) for pf in self.positive_filters) and
                not any(nf.program_priority_filter(p) for nf in self.negative_filters))

    @property
    def group_filter(self) -> Optional[GroupFilter]:
        return (lambda g: all(pf.group_filter(g) for pf in self.positive_filters) and
                not any(nf.group_filter(g) for nf in self.negative_filters))

    @property
    def group_priority_filter(self) -> Optional[GroupFilter]:
        return (lambda g: all(pf.group_priority_filter(g) for pf in self.positive_filters) and
                not any(nf.group_priority_filter(g) for nf in self.negative_filters))
