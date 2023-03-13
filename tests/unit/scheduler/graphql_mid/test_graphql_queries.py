# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
from scheduler.graphql_mid.server import  schema


def test_planpersite_query():


    query= """
        query plansPerSite{
            sitePlans(site: GS) {
                nightIdx
                plansPerSite {
                    site
                    startTime
                    endTime
                    visits {
                        startTime
                        obsId
                        atomStartIdx
                        atomEndIdx
                    }
                }
            }
        }
    """

    result = schema.execute_sync(query)
    assert result.errors is None
    # TODO: if we test for an exact solution right now as the Optimizer
    # is WIP these always would break. We need to revisit this test
    # when Gmax work is finished.
    # assert len(result.data["sitePlans"]) > 0


def test_schedule_query():

    query= """
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
    assert result.data["schedule"]["nightPlans"][0]["plansPerSite"][0]["startTime"] == "2018-10-01T04:59:00.000017+00:00"
