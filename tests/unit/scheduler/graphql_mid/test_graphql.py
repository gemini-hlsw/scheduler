# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import pytest

from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties

from scheduler.graphql_mid.server import schema


@pytest.mark.asyncio
async def test_schedule_query_required_only():
    ObservatoryProperties.set_properties(GeminiProperties)
    query = """
        query Schedule($programFile: Upload) {
            testSubQuery(scheduleId: "1", 
                         newScheduleInput: {startTime: "2018-10-01 08:00:00",
                                            endTime: "2018-10-03 08:00:00"
                                            sites: "GN", 
                                            mode: VALIDATION,
                                            semesterVisibility:False, 
                                            numNightsToSchedule:1,
                                            programFile:$programFile})
        }
    """
    result = await schema.execute(query)
    assert result.errors is None


@pytest.mark.asyncio
async def test_schedule_query_required_only():
    ObservatoryProperties.set_properties(GeminiProperties)
    query = """
        query Schedule($programFile: Upload) {
            testSubQuery(scheduleId: "1", 
                         newScheduleInput: {startTime: "2018-10-01 08:00:00",
                                            endTime: "2018-10-03 08:00:00"
                                            sites: "GN", 
                                            mode: VALIDATION,
                                            semesterVisibility:false, 
                                            numNightsToSchedule:1,
                                            programFile:$programFile})
        }
    """
    result = await schema.execute(query)
    assert result.errors is None

