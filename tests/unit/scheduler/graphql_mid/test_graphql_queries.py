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
    assert len(result.data["sitePlans"]) > 0
