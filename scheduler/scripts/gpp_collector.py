#!/usr/bin/env python3
# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, FrozenSet, Optional

import numpy as np
from astropy import units as u
from astropy.time import Time
from lucupy.minimodel import NightIndex, TimeslotIndex, VariantSnapshot
from lucupy.minimodel.constraints import CloudCover, ImageQuality
from lucupy.minimodel.semester import Semester
from lucupy.minimodel.site import ALL_SITES, Site
from lucupy.minimodel import Group, ObservationID, GroupID, UniqueGroupID, ProgramID, QAState, ObservationClass, ObservationStatus
from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties
from lucupy.timeutils import time2slots

from scheduler.core.builder.blueprint import CollectorBlueprint, SelectorBlueprint, OptimizerBlueprint
from scheduler.core.builder.validationbuilder import ValidationBuilder
from scheduler.core.builder.simulationbuilder import SimulationBuilder
from scheduler.core.components.ranker import RankerParameters, DefaultRanker
from scheduler.core.components.changemonitor import ChangeMonitor, TimeCoordinateRecord
from scheduler.core.eventsqueue.nightchanges import NightlyTimeline
from scheduler.core.output import print_collector_info, print_plans
from scheduler.core.plans import Plans
from scheduler.core.eventsqueue import EveningTwilightEvent, Event, EventQueue, MorningTwilightEvent, WeatherChangeEvent
from scheduler.core.sources.sources import Sources
from scheduler.core.sources.origins import Origins
from scheduler.services import logger_factory

import matplotlib.pyplot as plt

ObservatoryProperties.set_properties(GeminiProperties)

_logger = logger_factory.create_logger(__name__)

if __name__ == '__main__':
    # Initial variables
    verbose = False
    # Default
    start = Time("2024-11-01 08:00:00", format='iso', scale='utc')
    end = Time("2024-11-03 08:00:00", format='iso', scale='utc')
    num_nights_to_schedule = 1
    semester_visibility = False
    test_events = True
    sites = ALL_SITES
    ranker_parameters = RankerParameters()
    # ranker_parameters = RankerParameters(program_priority=10.0,
    #                                      preimaging_factor=1.25,
    #                                      # gs_altitude_limits={MinMax.MIN: Angle(18.0*u.deg), MinMax.MAX: Angle(75.0*u.deg)},
    #                                      # gn_altitude_limits={MinMax.MIN: Angle(40.0*u.deg), MinMax.MAX: Angle(88.0*u.deg)}
    #                                     )
    cc_per_site = None
    iq_per_site = None
    program_ids = None
    # ocs_program_ids = "/Users/bmiller/gemini/sciops/softdevel/Queue_planning/program_ids.txt"
    use_redis = False

    collector_blueprint = CollectorBlueprint(
        ['SCIENCE', 'PROGCAL', 'PARTNERCAL'],
        ['Q', 'LP', 'FT', 'DD', 'C', 'SV'],
        1.0
    )

    semesters = frozenset([Semester.find_semester_from_date(start.datetime),
                       Semester.find_semester_from_date(end.datetime)])

    if semester_visibility:
        end_date = max(s.end_date() for s in semesters)
        end_vis = Time(datetime(end_date.year, end_date.month, end_date.day).strftime("%Y-%m-%d %H:%M:%S"))
        diff = end - start + 1
        diff = int(diff.jd)
        night_indices = frozenset(NightIndex(idx) for idx in range(diff))
        num_nights_to_schedule = diff
    else:
        night_indices = frozenset(NightIndex(idx) for idx in range(num_nights_to_schedule))
        end_vis = end
        if not num_nights_to_schedule:
            raise ValueError("num_nights_to_schedule can't be None when visibility is given by end date")

    queue = EventQueue(night_indices, sites)
    sim_builder = SimulationBuilder(Sources(Origins.SIM.value()), queue)

    # Create the Collector, load the programs
    gpp_collector = sim_builder.build_collector(
        start=start,
        end=end_vis,
        num_of_nights=num_nights_to_schedule,
        sites=sites,
        semesters=semesters,
        with_redis=use_redis,
        blueprint=collector_blueprint,
        program_list='default'
    )

    print_collector_info(gpp_collector)

    p = gpp_collector.get_program(ProgramID('p-147'))
    print(p.id, p.internal_id, p.type)
    print(f"Start: {p.start}, End: {p.end}")
    p.show()

    obs = gpp_collector.get_observation(ObservationID('o-3b0'))
    targ = gpp_collector.get_target_info(obs.id)
    # plt.plot(targ[0].alt.degree)
    plt.plot(targ[0].airmass)
    plt.ylim(2.3, 0.95)
    plt.ylabel('Airmass')
    plt.show()

    print(obs.required_resources())

    for obs in p.observations():
        print(obs.id.id, obs.internal_id, obs.id.program_id.id)
        print(obs.exec_time())
        ti = gpp_collector.get_target_info(obs.id)
        print(ti[0].visibility_time.to(u.h))
        print(ti[0].rem_visibility_time.to(u.h))
        print(ti[0].rem_visibility_frac)

