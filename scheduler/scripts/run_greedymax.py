# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
from lucupy.minimodel import Site

from run_main import main
from scheduler.core.components.ranker import RankerParameters

if __name__ == '__main__':
    main(test_events=True,
         num_nights_to_schedule=3,
         ranker_parameters=RankerParameters(),
         verbose=False)
