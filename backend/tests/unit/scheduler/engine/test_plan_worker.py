# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""RT-24: worker-process entry points with a warm cached SCP.

PlanWorker is tested in-process with an injected build function; executor
integration is RT-25.
"""

import pickle
from datetime import datetime, timedelta, UTC
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from astropy import units as u
from astropy.coordinates import Angle
from astropy.time import Time
from lucupy.minimodel import CloudCover, ImageQuality, Site, VariantSnapshot

from scheduler.engine import plan_worker
from scheduler.engine.plan_worker import (SchedulerComputePayload, PlanWorker,
                                          worker_build, worker_compute)

NIGHT_START = datetime(2025, 3, 1, 23, 0, 0, tzinfo=UTC)
SITE = Site.GN


def make_scp():
    scp = MagicMock()
    scp.collector.time_slot_length.to_datetime.return_value = timedelta(minutes=1)
    scp.collector.night_events = {
        SITE: MagicMock(times=[[Time(NIGHT_START.replace(tzinfo=None), scale='utc')]])
    }
    plans = MagicMock()
    plans.night_idx = 0
    plans.plans = {SITE: SimpleNamespace(visits=[], night_stats=None, alt_degs=None)}
    scp.run_rt.return_value = plans
    return scp, plans


def make_variant():
    return VariantSnapshot(iq=ImageQuality.IQ70,
                           cc=CloudCover.CC50,
                           wind_dir=Angle(330.0, unit=u.deg),
                           wind_spd=5.0 * (u.m / u.s))


def make_payload(rt_event_factory, variants=None):
    return SchedulerComputePayload(
        event=rt_event_factory.on_demand(time=NIGHT_START + timedelta(minutes=30)),
        sites=frozenset([SITE]),
        night_times=None,
        variants=variants or {},
    )


def test_compute_requires_build(rt_event_factory):
    worker = PlanWorker(build_scp=MagicMock())
    assert not worker.is_built
    with pytest.raises(RuntimeError):
        worker.compute(make_payload(rt_event_factory))


def test_build_caches_scp_for_compute(rt_event_factory):
    scp, plans = make_scp()
    mock_build = MagicMock(return_value=scp)
    worker = PlanWorker(build_scp=mock_build)

    worker.build(MagicMock(), MagicMock())
    result1 = worker.compute(make_payload(rt_event_factory))
    result2 = worker.compute(make_payload(rt_event_factory))

    assert worker.is_built
    assert result1 is plans and result2 is plans
    mock_build.assert_called_once()


def test_build_replaces_cached_scp(rt_event_factory):
    scp_a, plans_a = make_scp()
    scp_b, plans_b = make_scp()
    worker = PlanWorker(build_scp=MagicMock(side_effect=[scp_a, scp_b]))

    worker.build(MagicMock(), MagicMock())
    assert worker.compute(make_payload(rt_event_factory)) is plans_a

    worker.build(MagicMock(), MagicMock())
    assert worker.compute(make_payload(rt_event_factory)) is plans_b


def test_compute_applies_variants(rt_event_factory):
    scp, _ = make_scp()
    variant = make_variant()
    worker = PlanWorker(build_scp=MagicMock(return_value=scp))

    worker.build(MagicMock(), MagicMock())
    worker.compute(make_payload(rt_event_factory, variants={SITE: variant}))

    scp.selector.update_site_variant.assert_called_once_with(SITE, variant)


def test_module_entry_points_share_one_worker(monkeypatch, rt_event_factory):
    """worker_build/worker_compute (what the executor invokes) must delegate
    to the same PlanWorker instance."""
    scp, plans = make_scp()
    monkeypatch.setattr(plan_worker, "_worker",
                        PlanWorker(build_scp=MagicMock(return_value=scp)))

    worker_build(MagicMock(), MagicMock())
    assert worker_compute(make_payload(rt_event_factory)) is plans


def test_compute_payload_is_picklable(rt_event_factory):
    payload = make_payload(rt_event_factory, variants={SITE: make_variant()})
    restored = pickle.loads(pickle.dumps(payload))

    assert restored.event == payload.event
    assert restored.sites == payload.sites
    assert restored.variants[SITE].iq == payload.variants[SITE].iq
