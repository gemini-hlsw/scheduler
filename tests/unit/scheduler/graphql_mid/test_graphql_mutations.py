# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
import pytest
from scheduler.graphql_mid.server import  schema


@pytest.mark.asyncio
async def test_newschedule_mutation():

    mutation = """
        mutation new_schedule {
            newSchedule(
                newScheduleInput: {startTime: "2018-10-01 08:00:00",
                                   endTime: "2018-10-03 08:00:00"}
            ) {
                __typename
                ... on NewScheduleSuccess {
                    success
                }
                ... on NewScheduleError {
                    error
                }
            }
        }
    """


    result = await schema.execute(mutation)

    assert result.errors is None
    assert result.data["newSchedule"]["success"]