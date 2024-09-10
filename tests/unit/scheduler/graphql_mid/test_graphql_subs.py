import pytest

from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties

from scheduler.graphql_mid.server import schema

@pytest.mark.asyncio
async def test_schedule_sub():
    ObservatoryProperties.set_properties(GeminiProperties)
    query = """
            query Schedule {
                testSubQuery(scheduleId: "1", 
                             newScheduleInput: {startTime: "2018-10-01 08:00:00",
                                                endTime: "2018-10-03 08:00:00"
                                                sites: "GN", 
                                                mode: VALIDATION,
                                                semesterVisibility:false,
                                                numNightsToSchedule:3})
            }
    """
    sub = """
         subscription MySubscription {
              queueSchedule(scheduleId:"1") {
                nightPlans {
                  nightTimeline {
                    nightIndex
                    timeEntriesBySite {
                      site
                      mornTwilight
                      eveTwilight
                      timeEntries {
                        startTimeSlots
                        event
                        plan {
                          nightConditions{
                            iq
                            cc
                          }
                          startTime
                          visits {
                            obsId
                            requiredConditions{
                              iq
                              cc
                            }
                          }
                          nightStats {
                            timeLoss
                            planScore
                            nToos
                            completionFraction
                            programCompletion
                          }
                        }
                      }
                    }
                  }
                }
                plansSummary
              }
            }   
    """
    sub_response = await schema.subscribe(sub)
    result = await schema.execute(query)
    print(result.data)
    print(sub_response)

