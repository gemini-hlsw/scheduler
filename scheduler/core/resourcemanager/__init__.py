# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from typing import Dict, Optional

from lucupy.minimodel import Resource

from scheduler.core.meta import Singleton


class ExternalService:
    """
    Use as common type to all external services used for the scheduler,
    regardless of Origin (GPP, OCS, Files, etc)
    """
    pass


class ResourceManager(ExternalService, metaclass=Singleton):
    """
    This is to avoid recreating repetitive resources.
    When we first get a resource ID string, create a Resource for it and store it here.
    Then fetch the Resources from here if they exist, and if they do not, then create a new one as per
    the lookup_resource method.
    """
    def __init__(self):
        self._all_resources: Dict[str, Resource] = {}

    def lookup_resource(self, resource_id: str) -> Optional[Resource]:
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
            self._all_resources[resource_id] = Resource(id=resource_id)
        return self._all_resources[resource_id]
