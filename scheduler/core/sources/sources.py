# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from io import BytesIO
from typing import FrozenSet

from lucupy.minimodel import Site

from scheduler.services.resource import FileResourceService
from .origins import Origin, Origins
from .services import Services


__all__ = [
    'Sources',
]


class Sources:
    """
    Sources provide the scheduler with the correct source info for each service.
    Default should be GPP connections. Other modes are OCS services and custom files.
    """

    def __init__(self, origin: Origin = Origins.OCS.value()):
        self.origin = None

    def set_origin(self, origin: Origin):
        self.origin = origin.load()

    def use_file(self,
                 sites: FrozenSet[Site],
                 service: Services,
                 calendar: BytesIO,
                 gmos_fpu: BytesIO,
                 gmos_gratings: BytesIO,
                 faults: BytesIO,
                 eng_tasks: BytesIO,
                 weather_closures: BytesIO) -> bool:

        match service:
            case Services.ENV:
                # Weather faults?
                return False

            case Services.RESOURCE:
                # Check that the amount of files is correct
                if gmos_fpu and gmos_gratings and faults and eng_tasks and weather_closures:
                    file_resource_service = FileResourceService()

                    for site in sites:
                        suffix = ('s' if site == Site.GS else 'n').upper()
                        file_resource_service.load_files(site,
                                                         f'GMOS{suffix}_fpu_barcode.txt',
                                                         gmos_fpu,
                                                         gmos_gratings,
                                                         faults,
                                                         eng_tasks,
                                                         weather_closures,
                                                         'telescope_schedules.xlsx')

                    self.set_origin(Origins.FILE.value())
                    self.origin.resource = file_resource_service
                    return True

                else:
                    raise ValueError('Missing files to load for service ')
