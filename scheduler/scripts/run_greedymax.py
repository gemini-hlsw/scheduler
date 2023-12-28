# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from lucupy.minimodel import ALL_SITES

from run_main import main
from scheduler.core.components.ranker import RankerParameters

if __name__ == '__main__':
    main(
        num_nights_to_schedule=3,
        sites=ALL_SITES,
        test_events=True,
        ranker_parameters=RankerParameters(),
        verbose=False
    )
