# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
import io

import pytest

from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties

from scheduler.graphql_mid.server import schema


@pytest.mark.asyncio
async def test_schedule_query_required_only(set_observatory_properties):

    query = """
        query Schedule {
            schedule(scheduleId: "1", 
                     newScheduleInput: {startTime: "2018-10-01 08:00:00",
                                        endTime: "2018-10-03 08:00:00"
                                        sites: "GN", 
                                        mode: VALIDATION,
                                        semesterVisibility:false, 
                                        numNightsToSchedule:1})
        }
    """
    result = await schema.execute(query)
    assert result.errors is None


@pytest.mark.asyncio
async def test_schedule_query_with_all(set_observatory_properties):
    query = """
        query Schedule($programFile: [String!]) {
            schedule(scheduleId: "1", 
                     newScheduleInput: {startTime: "2018-10-01 08:00:00",
                                        endTime: "2018-10-03 08:00:00"
                                        sites: "GN", 
                                        mode: VALIDATION,
                                        semesterVisibility:false, 
                                        numNightsToSchedule:1,
                                        thesisFactor: 2.1,
                                        power: 3,
                                        metPower: 2.334,
                                        visPower: 3.222,
                                        whaPower: 2.0,
                                        programs:$programFile})
        }
    """
    # Create a mock file
    # mock_file = io.BytesIO(b"GN-2018B-Q-101")
    # mock_file.name = "programs_ids.test.txt"
    program_list = []
    variables = {"file": program_list}

    result = await schema.execute(query, variable_values=variables)
    assert result.errors is None


@pytest.mark.asyncio
async def test_schedule_query_with_empty_file(set_observatory_properties):
    query = """
        query Schedule($programFile: [String!]) {
            schedule(scheduleId: "1", 
                     newScheduleInput: {startTime: "2018-10-01 08:00:00",
                                        endTime: "2018-10-03 08:00:00"
                                        sites: "GN", 
                                        mode: VALIDATION,
                                        semesterVisibility:false, 
                                        numNightsToSchedule:1,
                                        programs:$programFile})
        }
    """
    # Create a mock file
    # mock_file = io.BytesIO(b"")
    # mock_file.name = "programs_ids.test.txt"
    program_list = []
    variables = {"file": program_list}

    result = await schema.execute(query, variable_values=variables)
    assert result.errors is None


@pytest.mark.asyncio
async def test_schedule_query_with_wrong_parameters(set_observatory_properties):

    query = """
        query Schedule {
            schedule(scheduleId: "1", 
                     newScheduleInput: {startTime: "2018-10-01 08:00:00",
                                        endTime: "2018-10-03 08:00:00"
                                        sites: "GN", 
                                        mode: VALIDATION,
                                        semesterVisibility: false, 
                                        numNightsToSchedule:1,
                                        thesisFactor: 2.1,
                                        power: 3.2,
                                        metPower: 2.334,
                                        visPower: 3.222,
                                        whaPower: 2.0})
        }
    """

    result = await schema.execute(query)
    assert result.data is None
