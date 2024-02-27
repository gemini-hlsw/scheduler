# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from typing import Dict, TypeAlias

from lucupy.minimodel import NightIndex, Site, TimeslotIndex

# Alias for map from night index to time slot on which to start scoring a group or observation.
# The scores for all time slots prior to the night's time slot will be set to zero to allow for partial night scoring.
# Cannot include in Selector since this introduces a circular dependency.
StartingTimeslots: TypeAlias = Dict[Site, Dict[NightIndex, TimeslotIndex]]
