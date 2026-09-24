# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from datetime import date, datetime, timedelta
from typing import Any, Dict, final, FrozenSet, Final, Iterator, Optional, Tuple

from gql import Client
from gql.transport.requests import RequestsHTTPTransport
from lucupy.minimodel import ALL_SITES, Resource, ResourceType, Site

from scheduler.services import logger_factory
from .filters import CompositeFilter, ResourcesAvailableFilter
from .night_configuration import NightConfiguration
from .file_based_resource_service import FileBasedResourceService
from .telescope_nights_query import RESOURCE_URL, TELESCOPE_NIGHTS_QUERY

__all__ = [
    'OpsResourceService',
]


logger = logger_factory.create_logger(__name__)

# The resource service's `Instrument` enum tags that do not name a schedulable instrument:
# AO subsystems reported alongside instruments, and operational states/unknown runs.
_NON_INSTRUMENT_TAGS: Final[FrozenSet[str]] = frozenset({'ALTAIR', 'CANOPUS', 'ENGINEERING', 'UNKNOWN'})

# The resource service names instruments generically (e.g. 'GMOS' rather than 'GMOS-N'/'GMOS-S',
# 'F2' rather than 'Flamingos2') since it is not site-specific in its own tagging. Translate its
# tags to the Resource IDs used throughout the rest of the scheduler, mirroring
# GppProgramProvider._gpp_inst_to_ocs, which solves the same mismatch for ODB instrument names.
_INSTRUMENT_TAG_TO_RESOURCE_ID: Final[Dict[str, str]] = {
    'F2': 'Flamingos2',
    'GHOST': 'GHOST',
    'GNIRS': 'GNIRS',
    'GPI': 'GPI',
    'GSAOI': 'GSAOI',
    'IGRINS2': 'IGRINS-2',
    'MAROON_X': 'MAROON-X',
    'ALOPEKE': 'Alopeke',
    'CAL_ZORRO': 'Zorro',
    'NIRI': 'NIRI',
    'ACQ_CAM': 'ACQCAM',
    'IQUEYE': 'IQUEYE',
    'SCORPIO': 'SCORPIO',
    'GCAL': 'GCAL',
}

# The InstrumentComponentType values this service sources from the resource service, and the
# lucupy ResourceType each becomes. WFS and OTHER components are intentionally left alone for now:
# WFS availability still comes from the telescope schedule spreadsheet.
_COMPONENT_TYPE_TO_RESOURCE_TYPE: Final[Dict[str, ResourceType]] = {
    'FPU': ResourceType.FPU,
    'FILTER': ResourceType.FILTER,
    'DISPERSER': ResourceType.DISPERSER,
}

# The resource service's telescopeNights query rejects requests spanning more nights than this,
# so a wider [start, end) range must be split into chunks of at most this many nights each.
_MAX_NIGHTS_PER_QUERY: Final[int] = 400


@final
class OpsResourceService(FileBasedResourceService):
    """
    This is a mock for the future Resource service, used for the GPP operations (RT) mode
    It reads data regarding availability of instruments, IFUs, FPUs, MOS masks, etc. at each Site for given dates.

    It can then be queried to receive a set of Resource (usually with barcode IDs, except for instruments) for a
    given site on a given night.

    It caches and reuses Resources by ID as best as possible to minimize the number of Resource objects existing
    at any given time. Since Resource is immutable, this should be fine.

    Note that this is a Singleton class, so new instances do not need to be created.

    Unlike OcsResourceService and SimResourceService, instrument, FPU, filter, and disperser
    availability are not read from the telescope schedule spreadsheet or from FPU/filter/grating
    files: they are queried live from the resource service (the same `telescopeNights` GraphQL
    query used by ResourceEventSource) so that RT operations sees the currently mounted/usable set
    rather than a pre-published schedule. Everything else (WFS availability, faults, engineering
    tasks, weather closures, ToOs, and the LGS/ToO/blocked-night status that also comes out of the
    spreadsheet) is still file-based.
    """

    # Name of the spreadsheet file containing telescope configurations.
    _TEL_CALENDAR_FILE: Final[str] = 'telescope_schedules.xlsx'

    def __init__(self, sites: FrozenSet[Site] = ALL_SITES, subdir: str = 'operation'):
        """
        Create and initialize the Ops Resource object with the specified sites.
        """
        super().__init__(sites, subdir)

        for site in self._sites:
            suffix = ('s' if site == Site.GS else 'n').upper()

            self.load_files(site,
                            f'GMOS{suffix}_fpu_barcode.txt',
                            None,  # Per-night FPU availability comes from the resource service, not a file.
                            None,  # Per-night disperser availability comes from the resource service, not a file.
                            f'G{suffix}_faults.txt',
                            f'G{suffix}_engtasks.txt',
                            f'G{suffix}_weather_loss.txt',
                            None,  # Per-night filter availability comes from the resource service, not a file.
                            f'G{suffix}_toos.txt',
                            OpsResourceService._TEL_CALENDAR_FILE)

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

        # Replace the spreadsheet-derived instrument availability with live data from the resource
        # service, and add FPU, filter, and disperser availability, which were never loaded from a
        # file above.
        self._load_live_resources()

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

    @staticmethod
    def _instrument_resource_id(tag: str, site: Site) -> Optional[str]:
        """
        Translate a resource-service `Instrument` enum tag into the Resource ID used elsewhere in the
        scheduler, or None if the tag does not name a schedulable instrument (an AO subsystem or an
        operational state reported alongside instruments).
        """
        if tag == 'GMOS':
            return 'GMOS-N' if site == Site.GN else 'GMOS-S'
        if tag in _NON_INSTRUMENT_TAGS:
            return None
        resource_id = _INSTRUMENT_TAG_TO_RESOURCE_ID.get(tag)
        if resource_id is None:
            logger.warning(f'Unmapped resource-service instrument tag {tag!r}: using it as the Resource ID.')
            return tag
        return resource_id

    @staticmethod
    def _night_chunks(start: date, end: date) -> Iterator[Tuple[date, date]]:
        """
        Split the [start, end) night range into consecutive chunks of at most
        _MAX_NIGHTS_PER_QUERY nights, since the resource service rejects wider queries.
        """
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=_MAX_NIGHTS_PER_QUERY), end)
            yield chunk_start, chunk_end
            chunk_start = chunk_end

    def _load_live_resources(self) -> None:
        """
        Query the resource service for instrument, FPU, filter, and disperser availability over each
        site's date range, and use it in place of the spreadsheet/file-derived sets for those types.
        This blocks until the response is received: OpsResourceService can be constructed from within
        a running event loop (see night_monitor.orchestration), so this uses gql's synchronous requests
        transport rather than `execute_async`.
        """
        transport = RequestsHTTPTransport(url=RESOURCE_URL)
        client = Client(transport=transport)

        for site in self._sites:
            start = self._earliest_date_per_site[site]
            end = self._latest_date_per_site[site] + FileBasedResourceService._day

            # Drop the instrument resources the spreadsheet parsing added: RT operations gets these
            # live instead. FPU/filter/disperser resources were never loaded from a file, and WFS
            # resources loaded from the spreadsheet are left untouched. Also backfill an empty entry
            # for any date in range that _load_instrument_data never touched at all (a CLOSED/
            # ENGINEERING/SHUTDOWN night skips its row before recording any resources), since that
            # used to be filled in as a side effect of the now-skipped filter/grating CSV loading.
            d = start
            while d < end:
                self._resources[site][d] = {r for r in self._resources[site].get(d, set())
                                             if r.type != ResourceType.INSTRUMENT}
                d += FileBasedResourceService._day

            for chunk_start, chunk_end in self._night_chunks(start, end):
                result: Dict[str, Any] = client.execute(
                    TELESCOPE_NIGHTS_QUERY,
                    variable_values={
                        'site': site.name,
                        'start': chunk_start.isoformat(),
                        'end': chunk_end.isoformat(),
                    },
                )

                for night in result.get('telescopeNights', []):
                    night_date = datetime.strptime(night['observingNight'], '%Y-%m-%d').date()
                    if night_date < chunk_start or night_date >= chunk_end:
                        continue
                    resources = self._resources[site].setdefault(night_date, set())

                    for block in night.get('instrumentAvailability', []):
                        if block['usage'] == 'UNAVAILABLE':
                            continue
                        resource_id = self._instrument_resource_id(block['instrument'], site)
                        if resource_id is not None:
                            resource: Optional[Resource] = self.lookup_resource(resource_id,
                                                                                resource_type=ResourceType.INSTRUMENT)
                            if resource is not None:
                                resources.add(resource)

                    for block in night.get('components', []):
                        component = block['component']
                        resource_type = _COMPONENT_TYPE_TO_RESOURCE_TYPE.get(component['componentType'])
                        if block['usage'] == 'UNAVAILABLE' or resource_type is None:
                            continue
                        resource_id = component.get('barcode') or component['code']
                        resource = self.lookup_resource(resource_id,
                                                         description=component['name'],
                                                         resource_type=resource_type)
                        if resource is not None:
                            resources.add(resource)
