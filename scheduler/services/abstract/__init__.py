# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from scheduler.core.meta import Singleton


class ExternalService(metaclass=Singleton):
    """
    Use as common type to all external services used for the scheduler,
    regardless of Origin (GPP, OCS, Files, etc)
    """
    pass
