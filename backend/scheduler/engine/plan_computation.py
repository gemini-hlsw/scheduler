# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""Pure, synchronous plan computation for the real-time path.

Extracted from EngineRT._compute_event_plan and kept free of strawberry types
and event-loop state so it can run inside a worker process (RT-24/25). The
SPlans/NightPlansWithEvent conversion stays with the caller.
"""

import datetime
from typing import Dict, FrozenSet, Optional, Tuple

import numpy as np
from astropy.time import Time
from lucupy.minimodel import Site

from scheduler.core.events.queue.events import CustomStartOfNightEvent, Event
from scheduler.core.plans import NightStats, Plans
from scheduler.core.scp.scp import SCP
from scheduler.services import logger_factory

__all__ = ['compute_event_plans']

_logger = logger_factory.create_logger(__name__)


def compute_event_plans(scp: SCP,
                        sites: FrozenSet[Site],
                        event: Event,
                        night_times: Optional[Dict[Site, Tuple[Time, Time]]]) -> Plans:
    """Derive per-site start timeslots for the event, run the SCP, and
    post-process night stats and altitude data. Returns core Plans.

    The start timeslot for each site is:
    - the event timeslot if it happens after the twilight and any custom start,
    - the custom start timeslot if defined and the event happens before it,
    - the twilight timeslot (0) otherwise.
    """
    start_timeslot = {}
    for site in sites:
        night_start_time = scp.collector.night_events[site].times[0][0]
        utc_night_start = night_start_time.utc.to_datetime(timezone=datetime.timezone.utc)

        event_timeslot = event.to_timeslot_idx(
            utc_night_start,
            scp.collector.time_slot_length.to_datetime()
        )

        custom_start = night_times.get(site, (None, None))[0] if night_times else None
        if custom_start is not None:
            custom_start_timeslot = CustomStartOfNightEvent(
                site,
                custom_start.utc.to_datetime(timezone=datetime.timezone.utc),
                f"Custom start of night for site {site}"
            ).to_timeslot_idx(
                utc_night_start,
                scp.collector.time_slot_length.to_datetime()
            )
            site_timeslot = max(event_timeslot, custom_start_timeslot)
        else:
            site_timeslot = max(event_timeslot, 0)

        _logger.info(f"Computing plan for site {site.name} starting on timeslot: {site_timeslot}, "
                     f"utc_night_start={utc_night_start}, event_time={event.time}")
        start_timeslot[site] = {np.int64(0): site_timeslot}

    plans = scp.run_rt(start_timeslot)

    for site in sites:
        plans.plans[site].night_stats = NightStats({}, 0.0, 0, {}, {})
        plans.plans[site].alt_degs = []
        # Calculate altitude data
        for visit in plans.plans[site].visits:
            ti = scp.collector.get_target_info(visit.obs_id)
            end_time_slot = visit.start_time_slot + visit.time_slots
            values = ti[plans.night_idx].alt[visit.start_time_slot: end_time_slot]
            alt_degs = [val.dms[0] + (val.dms[1] / 60) + (val.dms[2] / 3600) for val in values]
            plans.plans[site].alt_degs.append(alt_degs)

    return plans
