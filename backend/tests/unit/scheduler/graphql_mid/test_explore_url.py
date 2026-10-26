# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""
Visits link to the observation's page in Explore.

Explore edits the ODB the scheduler reads, so the link must point at the Explore of
the same environment, and it must use the ODB ids (``p-…``, ``o-…``), not the
reference labels the plan shows.
"""

from datetime import datetime, UTC
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from gpp_client.environment import GPPEnvironment
from lucupy.minimodel import ObservationClass, ProgramID

from scheduler.clients import gpp as gpp_module
from scheduler.clients.gpp import explore_observation_url
from scheduler.core.plans import Visit
from scheduler.graphql_mid.types import SVisit

_PROGRAM_LABEL = ProgramID("G-2026B-0001")


@pytest.fixture
def environment(monkeypatch):
    """Pretend the installed gpp-client targets the given environment."""

    def _set(env: GPPEnvironment) -> None:
        monkeypatch.setattr(gpp_module, "GPPSettings", lambda: SimpleNamespace(environment=env))
        gpp_module._explore_base_url.cache_clear()

    yield _set
    gpp_module._explore_base_url.cache_clear()


@pytest.mark.parametrize(
    "env, host",
    [(GPPEnvironment.DEVELOPMENT, "explore-dev.gemini.edu"), (GPPEnvironment.PRODUCTION, "explore.gemini.edu")],
)
def test_url_follows_the_odb_environment(environment, env, host):
    environment(env)

    assert explore_observation_url("p-2a1", "o-1b3") == f"https://{host}/p-2a1/observation/o-1b3"


@pytest.mark.parametrize(
    "program_id, observation_id",
    [("GN-2018B-Q-101", "GN-2018B-Q-101-1"), ("p-2a1", None), (None, "o-1b3")],
)
def test_no_url_without_odb_ids(environment, program_id, observation_id):
    # OCS programs (validation mode) are not in the ODB, so Explore has nothing to show.
    environment(GPPEnvironment.PRODUCTION)

    assert explore_observation_url(program_id, observation_id) is None


def _visit() -> Visit:
    observation = SimpleNamespace(
        id="G-2026B-0001-Q-0001",
        internal_id="o-1b3",
        belongs_to=_PROGRAM_LABEL,
        targets=[],
        obs_class=ObservationClass.SCIENCE,
        constraints=SimpleNamespace(conditions=MagicMock()),
        instrument=lambda: None,
        fpu=lambda: None,
        disperser=lambda: None,
        filters=lambda: None,
    )
    return Visit(
        start_time=datetime(2026, 10, 23, 6, tzinfo=UTC),
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
        atom_times=[],
    )


def test_visit_links_through_the_program_odb_id(environment):
    environment(GPPEnvironment.DEVELOPMENT)
    programs = {_PROGRAM_LABEL: SimpleNamespace(internal_id="p-2a1")}

    s_visit = SVisit.from_computed_visit(_visit(), [], programs.get)

    assert s_visit.explore_url == "https://explore-dev.gemini.edu/p-2a1/observation/o-1b3"


def test_visit_without_a_program_lookup_has_no_link(environment):
    environment(GPPEnvironment.DEVELOPMENT)

    assert SVisit.from_computed_visit(_visit(), []).explore_url is None
    assert SVisit.from_computed_visit(_visit(), [], {}.get).explore_url is None
