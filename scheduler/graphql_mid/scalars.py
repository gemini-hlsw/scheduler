# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from typing import List, FrozenSet, Union, NewType

import strawberry  # noqa
from lucupy.minimodel import GroupID, ObservationID, ProgramID, Site, UniqueGroupID, ALL_SITES

from scheduler.config import ConfigurationError
from scheduler.core.sources import Origin, Origins


__all__ = [
    'Sites',
    'SObservationID',
    'SUniqueGroupID',
    'SGroupID',
    'SProgramID',
    'SOrigin',
]


def parse_sites(sites: Union[str, List[str]]) -> FrozenSet[Site]:
    """Parse Sites from Sites scalar

        Args:
            sites (Union[str, List[str]]): Option can be a list of sites or a single one

        Returns:
            FrozenSet[Site]: a frozen site that contains lucupy Site enums
                corresponding to each site.
        """

    def parse_specific_site(site: str):
        try:
            return Site[site]
        except KeyError:
            raise ConfigurationError('Missing site', site)

    if sites == 'ALL_SITES':
        # In case of ALL_SITES option, return lucupy alias for the set of all Site enums
        return ALL_SITES

    if isinstance(sites, list):
        return frozenset(map(parse_specific_site, sites))
    else:
        # Single site case
        return frozenset([parse_specific_site(sites)])


def parse_origins(name: str) -> Origin:
    print([o for o in Origins])
    try:
        return Origins[name].value()
    except KeyError:
        raise KeyError(f'Illegal origin specified: "{name}". Permitted values: {", ".join(o.value for o in Origins)}')


Sites = strawberry.scalar(NewType("Sites", FrozenSet[Site]),
                          description="Depiction of the sites that can be load to the collector",
                          serialize=lambda x: x,
                          parse_value=lambda x: parse_sites(x)) # noqa

SObservationID = strawberry.scalar(NewType('SObservationID', ObservationID),
                                   description='ID of an Observation',
                                   serialize=lambda x: x.id,
                                   parse_value=lambda x: ObservationID(x))

SUniqueGroupID = strawberry.scalar(NewType('SUniqueGroupID', UniqueGroupID),
                                   description='Unique ID of a Group',
                                   serialize=lambda x: x.id,
                                   parse_value=lambda x: UniqueGroupID(x))

SGroupID = strawberry.scalar(NewType('SGroupID', GroupID),
                             description='ID of an Group',
                             serialize=lambda x: x.id,
                             parse_value=lambda x: GroupID(x))

SProgramID = strawberry.scalar(NewType('SProgramID', ProgramID),
                               description='ID of an Program',
                               serialize=lambda x: x.id,
                               parse_value=lambda x: ProgramID(x))


SOrigin = strawberry.scalar(NewType('SOrigin', Origin),
                            description='Origin of the Source',
                            serialize=lambda x: str(x),
                            parse_value=lambda x: parse_origins(x))
