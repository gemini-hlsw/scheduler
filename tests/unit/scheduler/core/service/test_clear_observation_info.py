# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import os

from lucupy.minimodel.observation import ObservationClass, ObservationStatus
from lucupy.types import ZeroTime

from scheduler.core.builder.modes import ValidationMode
from scheduler.core.programprovider.ocs import read_ocs_zipfile, OcsProgramProvider
from scheduler.core.sources import Sources
from definitions import ROOT_DIR


def test_clear_observations():
    """
    Ensure the Validation clear_observation_info does as specified.
    """
    obs_classes = frozenset({ObservationClass.SCIENCE, ObservationClass.PROGCAL, ObservationClass.PARTNERCAL})
    sources = Sources()
    program_provider = OcsProgramProvider(obs_classes, sources)
    bad_status = frozenset([ObservationStatus.ONGOING, ObservationStatus.OBSERVED])

    # Read in a list of JSON data and parse into programs.
    program_data = read_ocs_zipfile(os.path.join(ROOT_DIR, 'scheduler', 'data', '2018B_program_samples.zip'))
    programs = [program_provider.parse_program(data['PROGRAM_BASIC']) for data in program_data]

    for program in programs:
        ValidationMode._clear_observation_info(program.observations(),  
                                               ValidationMode._obs_statuses_to_ready)

    # Check to make sure all data has been cleared.
    for p in programs:
        assert p.program_used() == ZeroTime
        assert p.partner_used() == ZeroTime
        assert p.total_used() == ZeroTime
        for o in p.observations():
            assert o.status not in bad_status
            assert o.program_used() == ZeroTime
            assert o.partner_used() == ZeroTime
            assert o.total_used() == ZeroTime
