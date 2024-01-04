# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from scheduler.graphql_mid.server import schema
from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties


def test_schedule_query():
    ObservatoryProperties.set_properties(GeminiProperties)
    query = """
        query schedule {
            schedule(
                newScheduleInput: {
                  startTime: "2018-10-01 08:00:00", 
                  numNightsToSchedule: 3, 
                  sites: "GN", 
                  mode: VALIDATION, 
                  endTime: "2018-10-03 08:00:00"}
            ) {
            nightPlans{
              nightTimeline{
                nightIndex
                timeEntriesBySite{
                  site,
                  timeEntries{
                    startTimeSlots,
                    event,
                    plan{
                      startTime,
                      visits{
                        obsId
                      },
                      nightStats{
                        timeloss
                      }
                    }
                  }
                }
              }
            },
            plansSummary
        }
    }
    """
    result = schema.execute_sync(query)
    assert result is not None
    result_data = result.data
    assert result_data is not None
    assert 'schedule' in result_data
    schedule = result_data['schedule']
    assert schedule is not None
    assert 'nightPlans' in schedule
    night_plans = schedule['nightPlans']
    assert night_plans is not None
    assert 'nightTimeline' in night_plans
    plan_per_night = night_plans['nightTimeline']
    assert plan_per_night is not None
    assert len(plan_per_night) >= 0
    time_entry = plan_per_night[0]
    assert time_entry is not None
    assert 'timeEntriesBySite' in time_entry
