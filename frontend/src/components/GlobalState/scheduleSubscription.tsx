import { graphql } from "@/gql";

export const subscriptionQueueSchedule = graphql(`
  subscription queueSchedule($scheduleId: String!) {
    queueSchedule(scheduleId: $scheduleId) {
      __typename
      ... on NewNightPlans {
        nightPlans {
          nightTimeline {
            nightIndex
            timeEntriesBySite {
              site
              mornTwilight
              eveTwilight
              timeEntries {
                startTimeSlots
                event {
                  time
                  site
                  description
                }
                plan {
                  startTime
                  nightConditions {
                    iq
                    cc
                  }
                  visits {
                    obsId
                    endTime
                    altitude
                    atomEndIdx
                    atomStartIdx
                    startTime
                    instrument
                    fpu
                    disperser
                    filters
                    score
                    obsClass
                    completion
                    atomTimes
                    peakScore
                    requiredConditions {
                      iq
                      cc
                    }
                  }
                  nightStats {
                    planScore
                    nToos
                    completionFraction
                    programCompletion
                  }
                }
                timelossWindows {
                  start
                  end
                  lossType
                }
                timestats {
                  nightLength
                  observed
                  scheduled
                  weather
                  fault
                  closed
                  unscheduled
                }
              }
            }
          }
        }
        plansSummary {
          summary
          metricsPerBand
        }
      }
      ... on NightPlansError {
        error
      }
      ... on NewPlansRT {
        nightPlans {
          nightIdx
          plansPerSite {
            endTime
            site
            startTime
            visits {
              altitude
              atomEndIdx
              atomStartIdx
              completion
              disperser
              endTime
              filters
              fpu
              instrument
              obsClass
              obsId
              peakScore
              score
              startTime
              requiredConditions {
                cc
                iq
              }
            }
            nightConditions {
              cc
              iq
            }
            nightStats {
              completionFraction
              nToos
              planScore
              programCompletion
            }
          }
        }
      }
      ... on NightPlansWithEvent {
        event
        nightPlans {
          nightIdx
          plansPerSite {
            endTime
            site
            startTime
            visits {
              altitude
              atomEndIdx
              atomStartIdx
              completion
              disperser
              endTime
              filters
              fpu
              instrument
              obsClass
              obsId
              peakScore
              score
              startTime
              requiredConditions {
                cc
                iq
              }
            }
            nightConditions {
              cc
              iq
            }
            nightStats {
              completionFraction
              nToos
              planScore
              programCompletion
            }
          }
        }
      }
    }
  }
`);
