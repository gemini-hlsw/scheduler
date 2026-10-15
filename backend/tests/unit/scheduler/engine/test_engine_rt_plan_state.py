# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""The stored plan must hold observations as the ODB has them, not as GreedyMax charged them.

GreedyMax marks everything it schedules ONGOING or OBSERVED on the Selector's copies. If the
plan kept those copies, no planned observation would ever be READY, and the Night Monitor's
READY checks against the plan would never match.
"""

from datetime import datetime, UTC
from types import SimpleNamespace
from unittest.mock import MagicMock

from lucupy.minimodel import ObservationID, ObservationStatus, Site

from scheduler.core.plans import Visit
from scheduler.engine.engineRT import EngineRT


def _observation(label: str, status: ObservationStatus) -> SimpleNamespace:
    return SimpleNamespace(id=ObservationID(label), status=status)


def _visit(observation) -> Visit:
    return Visit(
        start_time=datetime(2026, 10, 15, 6, 0, tzinfo=UTC),
        observation=observation,
        atom_start_idx=0,
        atom_end_idx=0,
        start_time_slot=0,
        time_slots=10,
        score=1.0,
        peak_score=1.0,
        step_start_idx=None,
        step_count=None,
        completion="1/1",
        atom_times=[10],
    )


def _engine(collector_observations) -> EngineRT:
    engine = EngineRT(
        MagicMock(),
        scheduler_queue=MagicMock(),
        process_id="test",
        weather_source=MagicMock(),
        nightly_timeline_store=MagicMock(),
    )
    engine.scp = MagicMock()
    engine.scp.collector.get_observation = lambda obs_id: collector_observations.get(obs_id)
    return engine


def test_visits_get_the_collector_observation_back():
    original = _observation("G-2026B-0001-Q-0001", ObservationStatus.READY)
    charged = _observation("G-2026B-0001-Q-0001", ObservationStatus.OBSERVED)
    visit = _visit(charged)
    engine = _engine({original.id: original})

    engine._drop_pseudo_time_accounting(SimpleNamespace(plans={Site.GN: SimpleNamespace(visits=[visit])}))

    assert visit.observation is original
    assert visit.observation.status is ObservationStatus.READY


def test_visit_missing_from_the_collector_keeps_its_copy():
    charged = _observation("G-2026B-0001-Q-0002", ObservationStatus.ONGOING)
    visit = _visit(charged)
    engine = _engine({})

    engine._drop_pseudo_time_accounting(SimpleNamespace(plans={Site.GS: SimpleNamespace(visits=[visit])}))

    assert visit.observation is charged
