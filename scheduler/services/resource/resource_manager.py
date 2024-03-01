# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from scheduler.core.meta import Singleton
from typing import final, Dict, Optional

from lucupy.minimodel import Resource, ResourceType


__all__ = [
    'ResourceManager',
]


@final
class ResourceManager(metaclass=Singleton):
    """
    A singleton class that manages Resource instances to reuse them as per the flyweight design pattern.
    """
    def __init__(self):
        """
        Create an empty dictionary of mappings from name to Resource.
        """
        self._all_resources: Dict[str, Resource] = {}

    def lookup_resource(self,
                        resource_id: str,
                        description: Optional[str] = None,
                        resource_type: Optional[ResourceType] = ResourceType.NONE) -> Optional[Resource]:
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
            self._all_resources[resource_id] = Resource(id=resource_id, description=description, type=resource_type)
        # Update description (e.g. MDF) if different from the original. For when the description can be changed.
        # if self._all_resources[resource_id].description != description:
        #     self._all_resources[resource_id].description = description
        return self._all_resources[resource_id]
