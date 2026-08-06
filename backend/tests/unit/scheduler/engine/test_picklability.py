# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""RT-23: everything that crosses the ProcessPoolExecutor boundary must
survive a pickle round-trip (worker payloads in, plan results out)."""

import pickle
from datetime import datetime, timedelta, UTC

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import Angle
from lucupy.minimodel import (CloudCover, Conditions, ImageQuality, NightIndex,
                              ObservationClass, ObservationID, Site, SkyBackground,
                              VariantSnapshot, WaterVapor)

from scheduler.core.plans import NightStats, Plan, Plans, Visit
from scheduler.engine.params import BuildParameters, NightTimes, SchedulerParameters

SITE = Site.GN


def roundtrip(obj):
    return pickle.loads(pickle.dumps(obj))


def test_scheduler_parameters_roundtrip():
    params = SchedulerParameters(start=datetime(2025, 3, 1, 8, 0, 0),
                                 end=datetime(2025, 3, 15, 8, 0, 0),
                                 semester_visibility=False,
                                 num_nights_to_schedule=1)
    restored = roundtrip(params)

    assert restored.start == params.start
    assert restored.end == params.end
    assert restored.sites == params.sites
    assert restored.mode == params.mode
    # RankerParameters holds numpy arrays, whose dataclass __eq__ raises
    # (ambiguous truth value), so compare fields explicitly.
    assert restored.ranker_parameters.thesis_factor == params.ranker_parameters.thesis_factor
    assert restored.ranker_parameters.power == params.ranker_parameters.power
    assert restored.ranker_parameters.met_power == params.ranker_parameters.met_power
    assert restored.ranker_parameters.vis_power == params.ranker_parameters.vis_power
    assert restored.ranker_parameters.wha_power == params.ranker_parameters.wha_power
    assert np.array_equal(restored.ranker_parameters.dec_diff, params.ranker_parameters.dec_diff)
    assert restored.ranker_parameters.score_combiner is params.ranker_parameters.score_combiner, \
        "score_combiner must pickle by reference (module-level function)"
    # Derived state from __post_init__ must survive too.
    assert restored.semesters == params.semesters
    assert restored.night_indices == params.night_indices
    assert restored.end_vis == params.end_vis


def test_build_parameters_roundtrip():
    build_params = BuildParameters(
        night_times={SITE: NightTimes(night_start=datetime(2025, 3, 1, 23, 0, 0),
                                      night_end=datetime(2025, 3, 2, 9, 30, 0))},
        visibility_start=datetime(2025, 3, 1, 8, 0, 0),
        visibility_end=datetime(2025, 3, 15, 8, 0, 0),
        program_list=["p-113"],
    )
    restored = roundtrip(build_params)

    assert restored == build_params
    # get_night_times() builds astropy Times from the restored data.
    (start, end) = restored.get_night_times()[SITE]
    assert start is not None and end is not None


def test_variant_snapshot_roundtrip():
    variant = VariantSnapshot(iq=ImageQuality.IQ70,
                              cc=CloudCover.CC50,
                              wind_dir=Angle(330.0, unit=u.deg),
                              wind_spd=5.0 * (u.m / u.s))
    restored = roundtrip(variant)

    assert restored.iq == variant.iq
    assert restored.cc == variant.cc
    assert restored.wind_dir == variant.wind_dir
    assert restored.wind_spd == variant.wind_spd


@pytest.mark.parametrize("factory_method", [
    "weather", "evening_twilight", "morning_twilight", "on_demand", "end_of_night",
])
def test_rt_events_roundtrip(rt_event_factory, factory_method):
    event = getattr(rt_event_factory, factory_method)()
    restored = roundtrip(event)

    assert restored == event  # UUID identity
    assert restored.site == event.site
    assert restored.time == event.time
    assert restored.description == event.description


def _make_plans() -> Plans:
    variant = VariantSnapshot(iq=ImageQuality.IQ70,
                              cc=CloudCover.CC50,
                              wind_dir=Angle(330.0, unit=u.deg),
                              wind_spd=5.0 * (u.m / u.s))
    plan = Plan(start=datetime(2025, 3, 1, 23, 0, 0, tzinfo=UTC),
                end=datetime(2025, 3, 2, 9, 30, 0, tzinfo=UTC),
                time_slot_length=timedelta(minutes=1),
                site=SITE,
                _time_slots_left=630,
                conditions=variant)
    plan.visits.append(Visit(start_time=datetime(2025, 3, 1, 23, 30, 0, tzinfo=UTC),
                             obs_id=ObservationID('G-2025A-Q-101-1'),
                             obs_class=ObservationClass.SCIENCE,
                             obs_conditions=Conditions(cc=CloudCover.CC50,
                                                       iq=ImageQuality.IQ70,
                                                       sb=SkyBackground.SB50,
                                                       wv=WaterVapor.WVANY),
                             atom_start_idx=0, atom_end_idx=1,
                             start_time_slot=30, time_slots=20,
                             score=10.0, peak_score=12.0,
                             step_start_idx=None, step_count=None,
                             instrument=None, fpu=None, disperser=None,
                             filters=None, completion='50%'))
    plan.night_stats = NightStats({}, 10.0, 0, {}, {})
    plan.alt_degs = [[45.5, 46.0]]

    plans = Plans(night_events={SITE: None},
                  night_conditions={SITE: variant},
                  night_idx=NightIndex(0))
    plans.plans[SITE] = plan
    return plans


def test_plan_result_roundtrip():
    plans = _make_plans()
    restored = roundtrip(plans)

    restored_plan = restored.plans[SITE]
    original_plan = plans.plans[SITE]

    assert restored.night_idx == plans.night_idx
    assert restored_plan.start == original_plan.start
    assert restored_plan.alt_degs == original_plan.alt_degs
    assert restored_plan.night_stats == original_plan.night_stats

    restored_visit = restored_plan.visits[0]
    assert restored_visit.obs_id == original_plan.visits[0].obs_id
    assert restored_visit.score == original_plan.visits[0].score
