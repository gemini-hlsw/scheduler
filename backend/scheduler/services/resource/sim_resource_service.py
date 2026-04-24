# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from datetime import date
from typing import final, FrozenSet, Final

from lucupy.minimodel import ALL_SITES, Site

from scheduler.services import logger_factory
from .filters import CompositeFilter, ResourcesAvailableFilter
from .night_configuration import NightConfiguration
from .file_based_resource_service import FileBasedResourceService

__all__ = [
    'SimResourceService',
]


logger = logger_factory.create_logger(__name__)


@final
class SimResourceService(FileBasedResourceService):
    """
    This is a mock for the future Resource service, used for the GPP simulation mode
    It reads data regarding availability of instruments, IFUs, FPUs, MOS masks, etc. at each Site for given dates.

    It can then be queried to receive a set of Resource (usually with barcode IDs, except for instruments) for a
    given site on a given night.

    It caches and reuses Resources by ID as best as possible to minimize the number of Resource objects existing
    at any given time. Since Resource is immutable, this should be fine.

    Note that this is a Singleton class, so new instances do not need to be created.
    """

    # Name of the spreadsheet file containing telescope configurations.
    _TEL_CALENDAR_FILE: Final[str] = 'telescope_schedules.xlsx'

    def __init__(self, sites: FrozenSet[Site] = ALL_SITES, subdir: str = 'simulation'):
        """
        Create and initialize the Sim Resource object with the specified sites.
        """
        super().__init__(sites, subdir)

        for site in self._sites:
            suffix = ('s' if site == Site.GS else 'n').upper()

            self.load_files(site,
                            f'GMOS{suffix}_fpu_barcode.txt',
                            f'GMOS{suffix}_FPUr202407.txt',
                            f'GMOS{suffix}_GRAT202407.txt',
                            f'G{suffix}_faults.txt',
                            f'G{suffix}_engtasks.txt',
                            f'G{suffix}_weather_loss.txt',
                            f'GMOS{suffix}_filters.txt',
                            f'G{suffix}_toos.txt',
                            SimResourceService._TEL_CALENDAR_FILE)

        # TODO: Remove this after discussion with science.
        # TODO: There are entries here outside of the Telescope Schedules Spreadsheet.
        # Record the earliest date for each site: any date before this will return an empty set of Resources.
        # Record the latest date for each site: any date after this will return the Resources on this date.
        # self._earliest_date_per_site = {site: min(self._resources[site], default=None) for site in self._sites}
        # self._latest_date_per_site = {site: max(self._resources[site], default=None) for site in self._sites}
        for site in self._sites:
            # Only one of these checks should be necessary.
            if self._earliest_date_per_site[site] == date.max or self._latest_date_per_site[site] == date.min:
                raise ValueError(f'No site resource data for {site.name}.')

        # Finalize the filters and create the night configurations.
        for site in self._sites:
            d = self._earliest_date_per_site[site]
            while d <= self._latest_date_per_site[site]:
                # Now that we have a complete set of resources per night:
                # 1. Make sure that there are entries in the positive_filters and negative_filters for the date.
                # 2. Add the ResourceFilter to the positive filters.
                # 2. Combine into a composite filter.
                pf = self._positive_filters[site].setdefault(d, set())
                nf = self._negative_filters[site].setdefault(d, set())
                pf.add(ResourcesAvailableFilter(frozenset(self._resources[site][d])))
                composite_filter = CompositeFilter(frozenset(pf), frozenset(nf))

                self._night_configurations[site][d] = NightConfiguration(
                    site=site,
                    local_date=d,
                    is_lgs=(d not in self._blocked[site] and self._lgs[site][d]),
                    too_status=(d not in self._blocked[site] and self._too[site][d]),
                    filter=composite_filter,
                    resources=frozenset(self._resources[site][d]),

                    # There may not be eng_tasks for the site or for the date at the site.
                    eng_tasks=frozenset(self._eng_tasks.get(site, {}).get(d, {}))
                )

                d += FileBasedResourceService._day
