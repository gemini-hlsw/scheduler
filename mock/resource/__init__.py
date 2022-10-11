# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import csv
import logging
import os
from datetime import date, datetime, timedelta
from types import MappingProxyType
from typing import Callable, Dict, FrozenSet, List, NoReturn, Set

from lucupy.helpers import str_to_bool
from lucupy.minimodel import ALL_SITES, Resource, Site
from openpyxl import load_workbook

from app.core.meta import Singleton
from definitions import ROOT_DIR
from .resources import Resources


class ResourceMock:
    # TODO: These are extra FPUS at GS that need to be added? They appear in the FPUr, so slight confusion.
    _extra_GS_fpus = ['11013104', '11013107', '11020601', '11022001', '11023313', '11023327', '11023328', '11023332',
                      '11023341', '11023342', '10000009', '10005373', '10005372', '10005374', '10005375', '10005376',
                      '10005390']

    def __init__(self, sites: FrozenSet[Site] = ALL_SITES):
        self._sites = sites
        self._path = os.path.join(ROOT_DIR, 'mock', 'resource', 'data')

        # This is to avoid recreating repetitive resources.
        # When we first get a resource ID string, create a Resource for it and store it here.
        # Then fetch the Resources from here if they exist, and if they do not, then create a new one as per
        # the _lookup_resource method.
        self._all_resources: Dict[str, Resource] = {}

        # The map from site and date to the set of resources.
        self._resources: Dict[Site, Dict[date, Set[Resource]]] = {site: {} for site in self._sites}

        # Mapping from ITCD FPUs to barcodes. The mapping is site-dependent.
        # The ODB program extractor produces long versions of these names that must be run through the
        # OcsFpuConverter to get the ITCD FPU names.
        self._fpu_to_barcode: Dict[Site, Dict[Resource, Resource]] = {}

        # Determines whether a night is a part of a laser run.
        self._lgs: Dict[Site, Dict[date, bool]] = {site: {} for site in self._sites}

    def connect(self):
        """
        Emulate a connection to a service and load the data.
        """
        for site in self._sites:
            # Load the mappings from the ITCD FPU values to the barcodes.
            self._load_fpu_to_barcodes(f'gmos{site.name}_fpu_barcode.txt')

            # TODO: Do we need this? As per discussion, this information is up to date in FPUr files.
            # Load the FPUs.
            # self._process_csv(site, f'GMOS{site.name}_FPU201789.txt', lambda r: {i.strip() for i in r[1:]})

            # Load the FPUrs.
            # This will put both the IFU and the FPU barcodes available on a given date as Resources.
            self._load_csv(site, f'GMOS{site.name}_FPUr201789.txt', lambda r: {i.strip() for i in r[1:]})

            # Load the gratings.
            # This will put MIRROR and the grating names available on a given date as Resources.
            self._load_csv(site, f'GMOS{site.name}_GRAT201789.txt',
                           lambda r: {'MIRROR'} | {i.strip().replace('+', '') for i in r[1:]})

        # Process the spreadsheet information for instrument, mode, and LGS settings.
        self._load_spreadsheet('2018B-2019A Telescope Schedules.xlsx')

    def _load_fpu_to_barcodes(self, site: Site, name: str) -> NoReturn:
        """
        FPUs at each site map to a unique barcode as defined in the:
            * gmos[ns]_fpu_barcode.txt
        files. These are site-dependent values.
        """
        with open(os.path.join(self._path, name)) as f:
            for row in f:
                fpu, barcode = row.split()

                # Only map if the FPU is a resource.
                if fpu in self._all_resources:
                    self._fpu_to_barcode[site][self._all_resources[fpu]] = self._lookup_resource(barcode)

    def _load_csv(self, site: Site, name: str, c: Callable[[List[str]], Set[str]]) -> NoReturn:
        """
        Process a CSV file as a table, where:
        1. The first entry is a date in YYYY-mm-dd format
        2. The remaining entries are resources available on that date.
        """
        with open(os.path.join(self._path, name)) as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                row_date = datetime.strptime(row[0].strip(), '%Y-%m-%d').date()
                date_set = self._resources[site].setdefault(row_date, set())
                date_set |= {self._lookup_resource(r) for r in c(row[1:])}
                self._resources[site][row_date] = date_set

    def _load_spreadsheet(self, name: str) -> NoReturn:
        """
        Process an Excel spreadsheet containing instrument, mode, and LGS information.
        """
        workbook = load_workbook(filename=os.path.join(self._path, name))
        for site in self._sites:
            sheet = workbook[site.name]

            # Read the sheet, skipping the header row (row 1).
            for row in sheet.iter_rows(min_row=2):
                # Skip shutdowns.
                mode = self._lookup_resource(row[1].value)
                if mode == 'Shutdown':
                    continue

                row_date = row[0].value.date()
                date_set = self._resources[site].setdefault(row_date, set())
                lgs = str_to_bool(row[2].value)
                instruments = {self._lookup_resource('Flamingos2' if c.value == 'F2' else c.value) for c in row[3:]}

                # TODO: Discuss with Bryan how to handle modes other than shutdowns?
                date_set |= instruments | {mode}
                self._resources[site][row_date] = date_set
                self._lgs[site][row_date] = lgs

    def _lookup_resource(self, resource_id: str) -> Resource:
        """
        Check if a Resource with id already exists.
        If it does, return it.
        If not, create it, add it to the list of all Resources, and then return it.
        """
        if resource_id not in self._all_resources:
            self._all_resources[resource_id] = Resource(id=resource_id)
        return self._all_resources[resource_id]

    @staticmethod
    def _previous(items, pivot: date):
        # Return date equal or previous to pivot
        tdmin = timedelta.min
        tdzero = timedelta()
        result = None

        for item in items:
            diff = item - pivot
            if tdzero >= diff > tdmin:
                result = item
                tdmin = diff
        return result

    def get_night_resources(self, sites: FrozenSet[Site], night_date: date) -> Dict[Site, FrozenSet[Resource]]:
        previous_date = ResourceMock._previous()
        return {site: frozenset(self._resources[site][night_date]) for site in sites}


class OcsFpuConverter(metaclass=Singleton):
    """
    These are the converters from the OCS FPU names to the ITCD FPU representations.
    For example, the ODB query extractor would return:
       * 'IFU 2 Slits'
    which we would want to convert to:
       * 'IFU-2'
    since these are the FPU names used in the GMOS[NS]-FPU(r?)######.txt files.
    """
    gmosn_ifu_dict = MappingProxyType({
        'IFU 2 Slits': 'IFU-2',
        'IFU Left Slit (blue)': 'IFU-B',
        'IFU Right Slit (red)': 'IFU-R',
        'Longslit 0.25 arcsec': '0.25arcsec',
        'Longslit 0.50 arcsec': '0.5arcsec',
        'Longslit 0.75 arcsec': '0.75arcsec',
        'Longslit 1.00 arcsec': '1.0arcsec',
        'Longslit 1.5 arcsec': '1.5arcsec',
        'Longslit 2.00 arcsec': '2.0arcsec',
        'Longslit 5.00 arcsec': '5.0arcsec',
        'N and S 0.50 arcsec': 'NS0.5arcsec',
        'N and S 0.75 arcsec': 'NS0.75arcsec',
        'N and S 1.00 arcsec': 'NS1.0arcsec',
        'N and S 1.50 arcsec': 'NS1.5arcsec',
        'N and S 2.00 arcsec': 'NS2.0arcsec',
        # TODO: Not in OCS?
        'focus_array_new': 'focus_array_new'
    })

    gmoss_ifu_dict = MappingProxyType({**gmosn_ifu_dict, **{
        'IFU N and S 2 Slits': 'IFU-NS-2',
        'IFU N and S Left Slit (blue)': 'IFU-NS-B',
        'IFU N and S Right Slit (red)': 'IFU-NS-R',
        # TODO: Not in OCS?
        'PinholeC': 'PinholeC'
    }})
