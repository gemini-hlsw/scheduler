# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from abc import ABC
from datetime import date
from typing import final

from lucupy.minimodel import Site


class ResourceServiceException(Exception, ABC):
    """
    Abstract superclass of exceptions raised when an issue is detected in the OCSResourceService.
    """
    def __init__(self, message: str):
        super().__init__(self, message)


@final
class SiteMissingException(ResourceServiceException):
    """
    Raised when there is a Site tab missing from the telescope configuration spreadsheet.
    """
    def __init__(self, service_name: str, site: Site):
        super().__init__(f'{service_name}: No tab for {site.name} in the '
                         'telescope schedule configuration spreadsheet.')


@final
class LocalDateMissingException(ResourceServiceException):
    """
    Raised when there is a gap in the telescope dates for a site.
    """
    def __init__(self, service_name: str, site: Site, missing_date: date):
        super().__init__(f'{service_name}: Site ${site.name} is missing data for date ${missing_date} '
                         'in the telescope configuration spreadsheet.')


@final
class UninstalledInstrumentAvailableException(ResourceServiceException):
    """
    Raised when an instrument is listed as Available or Engineering, but is not installed.
    TODO: These will change from instrument and status being str to being proper enumerated types.
    """
    def __init__(self, service_name: str, site: Site, discrepancy_date: date, instrument: str, status: str):
        super().__init__(f'{service_name}: Site {site.name} has instrument {instrument} listed as '
                         f'{status} yet not installed on a port on date {discrepancy_date}.')
