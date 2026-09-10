# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from enum import Enum
from typing import final


__all__ = [
    'Services',
]


@final
class Services(Enum):
    ENV = 'env'
    RESOURCE = 'resource'
