# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import os
from datetime import date
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import gelidum
from lucupy.minimodel import Site, ALL_SITES, Resource, ResourceType

from definitions import ROOT_DIR
from scheduler.services import logger_factory
from scheduler.services.abstract import ExternalService
from .event_generators import EngineeringTask, Fault
from .filters import *
from .night_configuration import NightConfiguration

__all__ = ['ResourceService']

logger = logger_factory.create_logger(__name__)


# Note: ExternalService makes this a Singleton.
class ResourceService(ExternalService):
    """
    This is to avoid recreating repetitive resources.
    When we first get a resource ID string, create a Resource for it and store it here.
    Then fetch the Resources from here if they exist, and if they do not, then create a new one as per
    the lookup_resource method.
    """
    # These are the converters from the OCS FPU names to the ITCD FPU representations.
    # For example, the ODB query extractor would return:
    #    * 'IFU 2 Slits'
    # which we would want to convert to:
    #    * 'IFU-2'
    # since these are the FPU names used in the GMOS[NS]-FPUr######.txt files.
    _gmosn_ifu_dict: Dict[str, str] = gelidum.freeze({
        'IFU 2 Slits': 'IFU-2',
        'IFU Left Slit (blue)': 'IFU-B',
        'IFU Right Slit (red)': 'IFU-R',
        'Longslit 0.25 arcsec': '0.25arcsec',
        'Longslit 0.50 arcsec': '0.5arcsec',
        'Longslit 0.75 arcsec': '0.75arcsec',
        'Longslit 1.00 arcsec': '1.0arcsec',
        'Longslit 1.50 arcsec': '1.5arcsec',
        'Longslit 2.00 arcsec': '2.0arcsec',
        'Longslit 5.00 arcsec': '5.0arcsec',
        'N and S 0.50 arcsec': 'NS0.5arcsec',
        'N and S 0.75 arcsec': 'NS0.75arcsec',
        'N and S 1.00 arcsec': 'NS1.0arcsec',
        'N and S 1.50 arcsec': 'NS1.5arcsec',
        'N and S 2.00 arcsec': 'NS2.0arcsec',
        'focus_array_new': 'focus_array_new'
    })

    _gmoss_ifu_dict: Dict[str, str] = gelidum.freeze({**_gmosn_ifu_dict, **{
        'IFU N and S 2 Slits': 'IFU-NS-2',
        'IFU N and S Left Slit (blue)': 'IFU-NS-B',
        'IFU N and S Right Slit (red)': 'IFU-NS-R',
        # TODO: Not in OCS? Correct, ENG observations that use this enter it as a Custom Mask
        'PinholeC': 'PinholeC'
    }})

    # Constants for converting MDF to barcodes.
    _instd = {'GMOS': '1', 'GMOS-N': '1', 'GMOS-S': '1', 'Flamingos2': '3'}
    _semd = {'A': '0', 'B': '1'}
    _progd = {'Q': '0', 'C': '1', 'L': '2', 'F': '3', 'S': '8', 'D': '9'}

    def __init__(self, sites: FrozenSet[Site] = ALL_SITES):
        self._all_resources: Dict[str, Resource] = {}
        self._sites = sites
        self._path = os.path.join(ROOT_DIR, 'scheduler', 'services', 'resource', 'data')

        # The map from site and date to the set of resources.
        self._resources: Dict[Site, Dict[date, Set[Resource]]] = {site: {} for site in self._sites}

        # Mapping from ITCD FPUs to barcodes. The mapping is site-dependent.
        # The ODB program extractor produces long versions of these names that must be run through the
        # OcsFpuConverter to get the ITCD FPU names.
        self._itcd_fpu_to_barcode: Dict[Site, Dict[str, Resource]] = {site: {} for site in self._sites}

        # Earliest and latest dates per site.
        self._earliest_date_per_site: Dict[Site, date] = {site: date.max for site in self._sites}
        self._latest_date_per_site: Dict[Site, date] = {site: date.min for site in self._sites}

        # Faults and engineering tasks.
        self._faults: Dict[Site, Dict[date, Set[Fault]]] = {site: {} for site in self._sites}
        self._eng_tasks: Dict[Site, Dict[date, Set[EngineeringTask]]] = {site: {} for site in self._sites}

        # Determines which nights are blocked.
        self._blocked: Dict[Site, Set[date]] = {site: set() for site in self._sites}

        # Determines whether a night is a part of a laser run.
        self._lgs: Dict[Site, Dict[date, bool]] = {site: {} for site in self._sites}

        # Determines whether ToOs are accepted on a given night.
        self._too: Dict[Site, Dict[date, bool]] = {site: {} for site in self._sites}

        # Filters to apply to a night. We add the ResourceFilters at the end after all the resources
        # have been processed.
        self._positive_filters: Dict[Site, Dict[date, Set[AbstractFilter]]] = {site: {} for site in self._sites}
        self._negative_filters: Dict[Site, Dict[date, Set[AbstractFilter]]] = {site: {} for site in self._sites}

        # The final combined filters for each night.
        self._filters: Dict[Site, Dict[date, AbstractFilter]] = {site: {} for site in self._sites}

        # The final output from this class: the configuration per night.
        self._night_configurations: Dict[Site, Dict[date, NightConfiguration]] = {site: {} for site in self._sites}

    def _mdf_to_barcode(self, mdf_name: str, inst: str) -> Optional[Resource]:
        """Legacy MOS mask barcode convention"""
        barcode = None
        if inst in ResourceService._instd.keys():
            # Collect the components of the string from the MDF name.
            inst_id = ResourceService._instd[inst]
            sem_id = ResourceService._semd[mdf_name[6]]
            progtype_id = ResourceService._progd[mdf_name[7]]
            barcode = f'{inst_id}{sem_id}{progtype_id}{mdf_name[-6:-3]}{mdf_name[-2:]}'
        return self.lookup_resource(barcode, description=mdf_name, type=ResourceType.FPU)

    def _itcd_fpu_to_barcode_parser(self, r: List[str], site: Site) -> Set[str]:
        return {self._itcd_fpu_to_barcode[site][r[0].strip()].id} | {i.strip() for i in r[1:]}

    def _convert_fpu_to_itcd_name(self, site: Site, fpu_name: str) -> Optional[str]:
        """
        Convert a long FPU name into its ITCD name, if it exists.
        """
        if Site.GN in self._sites and site == Site.GN:
            return self._gmosn_ifu_dict.get(fpu_name)
        if Site.GS in self._sites and site == Site.GS:
            return self._gmoss_ifu_dict.get(fpu_name)
        return None

    def lookup_resource(self, resource_id: str, description=None, type=ResourceType.NONE) -> Optional[Resource]:
        """
        Function to perform Resource caching and minimize the number of Resource objects by attempting to reuse
        Resource objects with the same ID.

        If resource_id evaluates to False, return None.
        Otherwise, check if a Resource with id already exists.
        If it does, return it.
        If not, create it, add it to the map of all Resources, and then return it.

        Note that even if multiple objects do exist with the same ID, they will be considered equal by the
        Resource equality comparator.
        """
        # The Resource constructor raises an exception for id None or containing any capitalization of "none".
        if not resource_id:
            return None
        if resource_id not in self._all_resources:
            self._all_resources[resource_id] = Resource(id=resource_id, description=description, type=type)
        # Update description (e.g. MDF) if different from the original. For when the description can be changed.
        # if self._all_resources[resource_id].description != description:
        #     self._all_resources[resource_id].description = description
        return self._all_resources[resource_id]

    def date_range_for_site(self, site: Site) -> Tuple[date, date]:
        """
        Return the date range (inclusive) for which we have resource data for a site.
        """
        if site not in self._sites:
            raise ValueError(f'Request for resource dates for illegal site: {site.name}')
        return self._earliest_date_per_site[site], self._latest_date_per_site[site]

    def get_night_configuration(self, site: Site, local_date: date) -> NightConfiguration:
        """
        Returns the NightConfiguration object for the site for the given local date,
        which contains the filters and resources for the night.
        """
        if site not in self._sites:
            raise ValueError(f'Request for night configuration for illegal site: {site.name}')
        if local_date < self._earliest_date_per_site[site] or local_date > self._latest_date_per_site[site]:
            raise ValueError(f'Request for night configuration for site {site.name} for illegal date: {local_date}')
        return self._night_configurations[site][local_date]

    def get_resources(self, site: Site, night_date: date) -> FrozenSet[Resource]:
        """
        For a site and a local date, return the set of available resources.
        The date is currently the truncation to day of the astropy Time objects in the time_grid, which are in UTC
           and have times of 8:00 am, so to get local dates, in the Collector we subtract one day.
        If the date falls before any resource data for the site, return the empty set.
        If the date falls after any resource data for the site, return the last resource set.
        """
        if site not in self._sites:
            raise ValueError(f'Request for resources for illegal site: {site.name}')

        # If the date is before the first date or after the last date, return the empty set.
        if night_date < self._earliest_date_per_site[site] or night_date > self._latest_date_per_site[site]:
            return frozenset()

        return frozenset(self._resources[site][night_date])

    def fpu_to_barcode(self, site: Site, fpu_name: str, instrument: str) -> Optional[Resource]:
        """
        Convert a long FPU name into the barcode, if it exists.
        """
        barcode = None
        itcd_fpu_name = self._convert_fpu_to_itcd_name(site, fpu_name)
        # print(f'fpu_to_barcode {fpu_name} {instrument} {itcd_fpu_name}')
        if itcd_fpu_name:
            barcode = self._itcd_fpu_to_barcode[site].get(itcd_fpu_name)
        elif fpu_name.startswith('G'):
            barcode = self._mdf_to_barcode(fpu_name, inst=instrument)
        # print(f'\t barcode {barcode.id} {barcode.description}')
        return barcode
