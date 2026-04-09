# First Steps

There are two ways to run the Scheduler the `run.py` script and the GraphQL playground.

## run.py Script

To run the script do:
```shell
$ python scheduler/scripts/run.py
```
That would run the default parameters inside the script to modify those you need to modify the [SchedulerParameters](architecture.md#scheduler-parameters).

## GraphQL Playground
To run the server do:
```shell
$ python scheduler/main.py
```

To access the Graphql playground you can go to `http://localhost:8000/graphql`. 

The Scheduler request work on two steps: the first one is a query sending the Scheduler Parameters and the request for a new plan and
the second one is subscription that outputs the results. Both of these need to be in separate tabs in the browser to work properly.

### Query Example

```
query Schedule {
                schedule(scheduleId: "1", 
                         newScheduleInput: {startTime: "2018-10-01 08:00:00",
                                            endTime: "2018-10-04 08:00:00"
                                            sites: "GN", 
                                            mode: SIMULATION,
                                            semesterVisibility:false,
                                            numNightsToSchedule:1})
            }

```

### Subscription Example

Check the ID is the same as the query, so the response is matched with the request.
```
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
      plansSummary{
        metricsPerBand
        summary
      }
    }
    ... on NightPlansError {
      error
    }
  }
}

```