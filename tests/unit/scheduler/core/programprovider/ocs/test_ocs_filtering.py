# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import os
from typing import FrozenSet, List

import pytest

from lucupy.minimodel import ObservationClass, Program

from scheduler.core.programprovider.ocs import read_ocs_zipfile, OcsProgramProvider
from definitions import ROOT_DIR


@pytest.fixture
def obs_classes() -> FrozenSet[ObservationClass]:
    return frozenset({ObservationClass.SCIENCE, ObservationClass.PROGCAL, ObservationClass.PARTNERCAL})


@pytest.fixture
def programs(obs_classes: FrozenSet[ObservationClass]) -> List[Program]:
    program_provider = OcsProgramProvider(obs_classes)
    program_data = read_ocs_zipfile(os.path.join(ROOT_DIR, 'scheduler', 'data', '2018B_program_samples.zip'))
    return [program_provider.parse_program(data['PROGRAM_BASIC']) for data in program_data]


def test_obsclass_filtering(programs: List[Program], obs_classes: FrozenSet[ObservationClass]):
    for program in programs:
        for observation in program.root_group.observations():
            assert observation.obs_class in obs_classes


def test_inactive_filtering(programs: List[Program]):
    for program in programs:
        for observation in program.root_group.observations():
            assert observation.active
