import pytest

from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties

from scheduler.graphql_mid.server import schema


@pytest.mark.asyncio
async def test_schedule_sub(visibility_calculator_fixture, set_observatory_properties):

    query = """
            query Schedule {
                schedule(scheduleId: "1", 
                         newScheduleInput: {startTime: "2018-10-01 08:00:00",
                                            endTime: "2018-10-04 08:00:00"
                                            sites: "GN", 
                                            mode: VALIDATION,
                                            semesterVisibility:false,
                                            numNightsToSchedule:3})
            }
    """
    sub = """
         subscription QueueSchedule {
          queueSchedule(scheduleId: "1") {
            __typename
            ... on NewNightPlans {
              nightPlans {
                nightTimeline {
                  nightIndex
                  timeEntriesBySite {
                    site
                    timeLosses
                    timeEntries {
                      event
                      plan {
                        startTime
                        endTime
                        nightConditions {
                          iq
                          cc
                        }
                        visits {
                          obsId
                        }
                        nightStats {
                          planScore
                          timeLoss
                        }
                      }
                    }
                  }
                }
              }
              plansSummary
            }
            ... on NightPlansError {
              error
            }
          }
        } 
    """
    sub_response = await schema.subscribe(sub)
    result = await schema.execute(query)

    async for result in sub_response:
        # Check return without errors
        assert not result.errors, 'Subscription returned with errors'
        # Check the correct number of nights.
        print([result.data["queueSchedule"]])
        n_nights = len(result.data["queueSchedule"]["nightPlans"]["nightTimeline"])
        print('Result night timeline: ', result.data["queueSchedule"]["nightPlans"]["nightTimeline"])
        assert n_nights == 1, f'Number of nights must be 3, but got {n_nights}'
        # Check plan summary is being calculated.
        assert result.data["queueSchedule"]["plansSummary"] is not None, 'Plans summary is not being calculated'
        # Check plan summary does not bring empty values
        assert any(v[0] != '0%' for v in result.data["queueSchedule"]["plansSummary"].values()), 'Plan summary is calculating empty programs'
        # Check that only one site is returned
        timeline = result.data["queueSchedule"]["nightPlans"]["nightTimeline"]
        assert any(len(night["timeEntriesBySite"]) == 1 for night in timeline), 'More than one site is returned'

        night = timeline[0]["timeEntriesBySite"][0]["timeEntries"][0]
        assert night["plan"]["nightConditions"] is not None, "Plan has missing weather conditions"
        assert night["plan"]["nightStats"] is not None, "Night stats were not calculated"
        assert night["plan"]["nightStats"]["planScore"] > 0, "Plan score is zero or negative value"

        break
