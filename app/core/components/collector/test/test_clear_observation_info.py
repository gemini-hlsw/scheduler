# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from datetime import timedelta
import os

from lucupy.minimodel.observation import ObservationStatus

from app.core.components.collector import Collector
from app.core.programprovider.ocs import read_ocs_zipfile, OcsProgramProvider
from definitions import ROOT_DIR


def test_clear_observations():
    """
    Ensure the Collector clear_observation_info does as specified.
    """
    program_provider = OcsProgramProvider()
    bad_status = frozenset([ObservationStatus.ONGOING, ObservationStatus.OBSERVED])
    zero = timedelta()

    # Read in a list of JSON data and parse into programs.
    program_data = read_ocs_zipfile(os.path.join(ROOT_DIR, 'app', 'data', '2018B_program_samples.zip'))
    programs = [program_provider.parse_program(data['PROGRAM_BASIC']) for data in program_data]

    # Clear the observations.
    Collector.clear_observation_info(programs)

    # Check to make sure all data has been cleared.
    for program in programs:
        for o in program.observations():
            assert o.status not in bad_status
            assert o.program_used() == zero
            assert o.partner_used() == zero
