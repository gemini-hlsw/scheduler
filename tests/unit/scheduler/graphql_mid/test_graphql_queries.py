# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
from scheduler.graphql_mid.server import schema
from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini import GeminiProperties

def test_schedule_query():
    ObservatoryProperties.set_properties(GeminiProperties)
    query = """
        query getNightPlans {
                schedule(newScheduleInput: {startTime: "2018-10-01 08:00:00",
    				                        endTime: "2018-10-03 08:00:00",
    				                        site: "ALL_SITES"}) {
                nightPlans {
                    nightIdx
                    plansPerSite {
                        endTime
                        site
                        startTime
                        visits {
                            atomEndIdx
                            atomStartIdx
                            obsId
                            startTime
                        }
                    }
                }
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
    assert len(night_plans) >= 1
    night_plan = night_plans[0]
    assert night_plan is not None
    assert 'plansPerSite' in night_plan
    plan_per_site = night_plan['plansPerSite']
    assert plan_per_site is not None
    assert len(plan_per_site) >= 0
    plan = plan_per_site[0]
    assert plan is not None
