# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import csv
import logging
import os
from datetime import date, datetime, timedelta
from typing import Callable, Dict, FrozenSet, List, NoReturn, Set, Tuple

from lucupy.helpers import str_to_bool
from lucupy.minimodel import ALL_SITES, Resource, Site
from openpyxl import load_workbook

from definitions import ROOT_DIR
from .resources import Resources


class ResourceMock:
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

        # FPU to barcodes. The mapping is site-dependent.
        self._fpu_to_barcode: Dict[Site, Dict[Resource, Resource]] = {}

        # Determines whether a night is a part of a laser run.
        self._lgs: Dict[Site, Dict[date, bool]] = {site: {} for site in self._sites}

    def connect(self):
        """
        Emulate a connection to a service and load the data.
        """
        for site in self._sites:
            # Load the FPUs.
            self._process_csv(site, f'GMOS{site.name}_FPU201789.txt', lambda r: {i.strip() for i in r[1:]})

            # Load the barcodes.
            self._process_fpu_to_barcodes(f'gmos{site.name}_fpu_barcode.txt')

            # Load the FPUrs.
            self._process_csv(site, f'GMOS{site.name}_FPUr201789.txt', lambda r: {i.strip() for i in r[1:]})

            # Load the gratings.
            self._process_csv(site, f'GMOS{site.name}_GRAT201789.txt',
                              lambda r: {'MIRROR'} | {i.strip().replace('+', '') for i in r[1:]})

        # Process the spreadsheet information for instrument, mode, and LGS settings.
        self._process_spreadsheet('2018B-2019A Telescope Schedules.xlsx')

    def _process_csv(self, site: Site, name: str, c: Callable[[List[str]], Set[str]]) -> NoReturn:
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

    def _process_spreadsheet(self, name: str) -> NoReturn:
        """
        Process an Excel spreadsheet containing instrument, mode, and LGS information.
        """
        workbook = load_workbook(filename=os.path.join(self._path, name))
        for site in self._sites:
            sheet = workbook[site.name]
            for row in sheet.iter_rows(min_row=2):
                # TODO: Check this to make sure it is a date, and if not, convert.
                row_date = row[0].value
                date_set = self._resources[site].setdefault(row_date, set())
                mode = self._lookup_resource(row[1].value)
                lgs = str_to_bool(row[2].value)
                instruments = {self._lookup_resource('Flamingos2' if c.value == 'F2' else c.value) for c in row[3:]}

                # TODO: Discuss with Bryan how to handle mode?
                date_set |= instruments | {mode}
                self._resources[site][row_date] = date_set
                self._lgs[site][row_date] = lgs

    def _lookup_resource(self, id: str) -> Resource:
        """
        Check if a Resource with id already exists.
        If it does, return it.
        If not, create it, add it to the list of all Resources, and then return it.
        """
        if id not in self._all_resources:
            self._all_resources[id] = Resource(id=id)
        return self._all_resources[id]

    def _process_fpu_to_barcodes(self, site: Site, name: str) -> NoReturn:
        """
        FPUs at each site map to a unique barcode. These are site-dependent values.
        """
        with open(os.path.join(self._path, name)) as f:
            for row in f:
                fpu, barcode = row.split()

                # Only map if the FPU is a resource.
                if fpu in self._all_resources:
                    self._fpu_to_barcode[site][self._all_resources[fpu]] = self._lookup_resource(barcode)

    @staticmethod
    def _previous(items, pivot):
        # Return date equal or previous to pivot
        tdmin = timedelta.min
        tdzero = timedelta(days=0)
        result = None

        for item in items:
            diff = item - pivot
            if tdzero >= diff > tdmin:
                result = item
                tdmin = diff
        return result

    def _get_info(self, site: Site, resource_date: str):
        info_types = {'fpu': self.fpu[site],
                      'fpur': self.fpur[site],
                      'grat': self.grat[site],
                      'instr': self.instruments[site],
                      'LGS': self.lgs[site],
                      'mode': self.mode[site],
                      'fpu-ifu': self.ifu[site]['FPU'],
                      'fpur-ifu': self.ifu[site]['FPUr']}

        if info in info_types:
            previous_date = ResourceMock._previous(info_types[info].keys(), date)
            return info_types[info][previous_date]

        else:
            logging.warning(f'No information about {info} is stored')
            return None

    def get_night_resources(self, sites: FrozenSet[Site], night_date: date) -> Dict[Site, FrozenSet[Resource]]:
        return {site: frozenset(self._resources[site][night_date]) for site in sites}
        # for site in sites:
        #     night_resources[site] = self._resources[site][night_date]
        # return self._reso
        # def night_info(info_name: str):
        #     return {site: self._get_info(info_name, site, night_date) for site in sites}
