# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from typing import final, FrozenSet

from lucupy.minimodel import ALL_SITES, Site

from .file_based_resource_service import FileBasedResourceService

__all__ = ['FileResourceService']


@final
class FileResourceService(FileBasedResourceService):
    def __init__(self, sites: FrozenSet[Site] = ALL_SITES):
        # Init an empty ResourceService
        super().__init__(sites)
