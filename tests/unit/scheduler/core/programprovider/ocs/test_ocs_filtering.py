# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from typing import FrozenSet, List

import pytest

from lucupy.minimodel import ObservationClass, Program

from scheduler.core.programprovider.ocs import ocs_program_data, OcsProgramProvider
from scheduler.core.sources import Sources


@pytest.fixture
def obs_classes() -> FrozenSet[ObservationClass]:
    return frozenset({ObservationClass.SCIENCE, ObservationClass.PROGCAL, ObservationClass.PARTNERCAL})


@pytest.fixture
def programs(obs_classes: FrozenSet[ObservationClass]) -> List[Program]:
    sources = Sources()
    program_provider = OcsProgramProvider(obs_classes, sources)
    program_data = ocs_program_data()
    return [program_provider.parse_program(data['PROGRAM_BASIC']) for data in program_data]


def test_obsclass_filtering(programs: List[Program], obs_classes: FrozenSet[ObservationClass]):
    for program in programs:
        for observation in program.root_group.observations():
            assert observation.obs_class in obs_classes


def test_inactive_filtering(programs: List[Program]):
    for program in programs:
        for observation in program.root_group.observations():
            assert observation.active
