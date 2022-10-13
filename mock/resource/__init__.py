# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from copy import copy
import csv
import os
from datetime import date, datetime, timedelta
from types import MappingProxyType
from typing import Callable, Dict, FrozenSet, List, NoReturn, Optional, Set

from lucupy.helpers import str_to_bool
from lucupy.minimodel import ALL_SITES, Resource, Site
from openpyxl import load_workbook

from app.core.meta import Singleton
from definitions import ROOT_DIR


# TODO: Should this be a Singleton?
class ResourceMock(metaclass=Singleton):
    # TODO: These are extra FPUS at GS that need to be added? They appear in the FPUr, so slight confusion.
    _extra_GS_fpus = ['11013104', '11013107', '11020601', '11022001', '11023313', '11023327', '11023328', '11023332',
                      '11023341', '11023342', '10000009', '10005373', '10005372', '10005374', '10005375', '10005376',
                      '10005390']

    # Definition of a day to not have to redeclare constantly.
    _day = timedelta(days=1)

    def __init__(self, sites: FrozenSet[Site] = ALL_SITES):
        self._sites = sites
        self._path = os.path.join(ROOT_DIR, 'mock', 'resource', 'data')

        # This is to avoid recreating repetitive resources.
        # When we first get a resource ID string, create a Resource for it and store it here.
        # Then fetch the Resources from here if they exist, and if they do not, then create a new one as per
        # the _lookup_resource method.
        # Initialize with the _extra_GS_fpus if GS is included.
        self._all_resources: Dict[str, Resource] = {} if Site.GS not in sites \
            else {gs_fpu: Resource(id=gs_fpu) for gs_fpu in ResourceMock._extra_GS_fpus}

        # The map from site and date to the set of resources.
        self._resources: Dict[Site, Dict[date, Set[Resource]]] = {site: {} for site in self._sites}

        # Mapping from ITCD FPUs to barcodes. The mapping is site-dependent.
        # The ODB program extractor produces long versions of these names that must be run through the
        # OcsFpuConverter to get the ITCD FPU names.
        self._fpu_to_barcode: Dict[Site, Dict[str, Resource]] = {site: {} for site in self._sites}

        # Determines whether a night is a part of a laser run.
        self._lgs: Dict[Site, Dict[date, bool]] = {site: {} for site in self._sites}

    def connect(self):
        """
        Emulate a connection to a service and load the data.
        """
        def suffix(site: Site, uc: bool = False) -> str:
            sfx = 'n' if site == Site.GN else 's'
            return sfx.upper() if uc else sfx

        for site in self._sites:
            # Load the mappings from the ITCD FPU values to the barcodes.
            self._load_fpu_to_barcodes(site, f'gmos{suffix(site)}_fpu_barcode.txt')

            # TODO: Do we need this? As per discussion, this information is up to date in FPUr files.
            # Load the FPUs.
            # self._process_csv(site, f'GMOS{site.name}_FPU201789.txt', lambda r: {i.strip() for i in r[1:]})

            # Load the FPUrs.
            # This will put both the IFU and the FPU barcodes available on a given date as Resources.
            # Note that for the IFU, we need to convert to a barcode, which is a Resource.
            # This is a bit problematic since we expect a list of strings of Resource IDs, so we have to take
            # its ID.
            self._load_csv(site, f'GMOS{"N" if site == Site.GN else "S"}_FPUr201789.txt',
                           lambda r: {self._fpu_to_barcode[site][r[0].strip()].id} | {i.strip() for i in r[1:]})

            # Load the gratings.
            # This will put the mirror and the grating names available on a given date as Resources.
            # TODO: Check Mirror vs. MIRROR. Seems like GMOS uses Mirror.
            self._load_csv(site, f'GMOS{"N" if site == Site.GN else "S"}_GRAT201789.txt',
                           lambda r: {'Mirror'} | {i.strip().replace('+', '') for i in r})

        # Process the spreadsheet information for instrument, mode, and LGS settings.
        self._load_spreadsheet('2018B-2019A Telescope Schedules.xlsx')

        # Add the GS FPUs to every date if GS is included in the sites.
        if Site.GS in self._sites:
            gs_fpus = frozenset([self._lookup_resource(fpu_name) for fpu_name in ResourceMock._extra_GS_fpus])
            for res_date, data in self._resources[Site.GS].items():
                self._resources[Site.GS][res_date] = data | gs_fpus

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
                if fpu is not None:
                    self._fpu_to_barcode[site][fpu] = self._lookup_resource(barcode)

    def _load_csv(self, site: Site, name: str, c: Callable[[List[str]], Set[str]]) -> NoReturn:
        """
        Process a CSV file as a table, where:

        1. The first entry is a date in YYYY-mm-dd format
        2. The remaining entries are resources available on that date to the following date in the file.

        If a date is missing from the CSV file, copy the data from the previously defined date through to just before
        the new date.
        """
        with open(os.path.join(self._path, name)) as f:
            reader = csv.reader(f, delimiter=',')
            prev_row_date: Optional[date] = None
            for row in reader:
                row_date = datetime.strptime(row[0].strip(), '%Y-%m-%d').date()

                # Fill in any gaps by copying prev_row_date until we reach one less than row_date.
                if prev_row_date is not None:
                    missing_row_date = prev_row_date + ResourceMock._day
                    while missing_row_date < row_date:
                        # Make sure there is an entry and append to it to avoid overwriting anything already present.
                        date_set = self._resources[site].setdefault(missing_row_date, set())
                        date_set |= copy(self._resources[site][prev_row_date])
                        self._resources[site][missing_row_date] = date_set
                        missing_row_date += ResourceMock._day

                # Create a set to append to if one does not exist.
                date_set = self._resources[site].setdefault(row_date, set())

                # Filter out any empty entries in the row.
                new_entries = {self._lookup_resource(r) for r in c(row[1:]) if r}
                self._resources[site][row_date] = date_set | new_entries

                # Advance the previous row date where data was defined.
                prev_row_date = row_date

    def _load_spreadsheet(self, name: str) -> NoReturn:
        """
        Process an Excel spreadsheet containing instrument, mode, and LGS information.

        The Excel spreadsheets have information available for every date, so we do not have to concern ourselves
        as in the _load_csv file above.
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

                # Filter out any ports that are empty.
                instruments = {self._lookup_resource('Flamingos2' if c.value == 'F2' else c.value)
                               for c in row[3:] if c.value is not None}

                # TODO: Discuss with Bryan how to handle modes other than shutdowns?
                # TODO: It doesn't seem correct to include the mode in the Resource set.
                date_set |= instruments | {mode}
                self._resources[site][row_date] = date_set
                self._lgs[site][row_date] = lgs

    def _lookup_resource(self, resource_id: str) -> Resource:
        """
        Check if a Resource with id already exists.
        If it does, return it.
        If not, create it, add it to the list of all Resources, and then return it.
        """
        if not resource_id:
            raise ValueError('Empty resource ID specified')
        if resource_id not in self._all_resources:
            self._all_resources[resource_id] = Resource(id=resource_id)
        return self._all_resources[resource_id]

    def get_night_resources(self, sites: FrozenSet[Site], night_date: date) -> Dict[Site, FrozenSet[Resource]]:
        missing_sites = [site for site in sites if site not in self._sites]
        if missing_sites:
            raise ValueError(f'Illegal sites for ResourceMock: accepted={self._sites}, requested={missing_sites}')
        return {site: frozenset(self._resources[site][night_date] if night_date in self._resources[site] else ())
                for site in sites if site in self._sites}

    def fpu_to_barcode(self, site: Site, fpu_name: str) -> Optional[Resource]:
        """
        Convert a long FPU name into the barcode, if it exists.
        """
        return self._fpu_to_barcode[site].get(fpu_name)

    # # TODO: **************************************************************
    # # TODO: We are not using masks yet: just FPUs and dispersers for GMOS.
    # # TODO: **************************************************************
    # _decoder = {'A': '0', 'B': '1', 'Q': '0',
    #             'C': '1', 'LP': '2', 'FT': '3',
    #             'SV': '8', 'DD': '9'}
    # _pattern = '|'.join(_decoder.keys())
    #
    # @staticmethod
    # def _decode_mask(mask_name: str) -> str:
    #     return '1' + re.sub(f'({ResourceMock._pattern})',
    #                         lambda m: ResourceMock._decoder[m.group()], mask_name).replace('-', '')[6:]
    #
    # def is_mask_available(self, site: Site, fpu_mask: str) -> bool:
    #     barcode = None
    #     if fpu_mask == 'None':
    #         return True
    #     if fpu_mask in self._fpu_to_barcode[site]:
    #         barcode = self._fpu_to_barcode[site][self._lookup_resource(fpu_mask)]
    #     elif '-' in fpu_mask:
    #         barcode = ResourceMock._decode_mask(fpu_mask)
    #
    #     return barcode and barcode in self._resources[site]

    """
    These are the converters from the OCS FPU names to the ITCD FPU representations.
    For example, the ODB query extractor would return:
       * 'IFU 2 Slits'
    which we would want to convert to:
       * 'IFU-2'
    since these are the FPU names used in the GMOS[NS]-FPU(r?)######.txt files.
    """
    _gmosn_ifu_dict = MappingProxyType({
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

    _gmoss_ifu_dict = MappingProxyType({**_gmosn_ifu_dict, **{
        'IFU N and S 2 Slits': 'IFU-NS-2',
        'IFU N and S Left Slit (blue)': 'IFU-NS-B',
        'IFU N and S Right Slit (red)': 'IFU-NS-R',
        # TODO: Not in OCS?
        'PinholeC': 'PinholeC'
    }})

    def _convert_fpu_to_itcd_name(self, site: Site, fpu_name: str) -> Optional[str]:
        if Site.GN in self._sites and site == Site.GN:
            return self._gmosn_ifu_dict.get(fpu_name)
        if Site.GS in self._sites and site == Site.GS:
            return self._gmoss_ifu_dict.get(fpu_name)
        return None

    def convert_fpu_to_barcode(self, site: Site, fpu_long_name: str) -> Optional[Resource]:
        """
        Convert a long FPU name in GMOS to the Resource consisting of a barcode.
        """
        itcd_name = self._convert_fpu_to_itcd_name(site, fpu_long_name)
        return None if itcd_name is None else self.fpu_to_barcode(site, itcd_name)

    def lookup_resource(self, resource_name: str) -> Optional[Resource]:
        """
        Given a resource name, look it up and retrieve the Resource object from the cache if it exists.
        If not, None is returned.
        """
        return self._all_resources.get(resource_name)


# For Bryan: testing
if __name__ == '__main__':
    # To get the Resources for a specific site on a specific date, modify the following:
    site = Site.GN
    day = date(year=2018, month=11, day=30)

    r = ResourceMock()
    r.connect()
    resources_available = r.get_night_resources(frozenset([site]), day)
    print(f'*** Resources for site {site.name} for {day} ***')
    for resource in sorted(resources_available[site], key=lambda x: x.id):
        print(resource)
