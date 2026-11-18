import { graphql } from "@/gql";

// The plans of a night timeline, shared by the plan in effect and a proposed one.
export const nightPlansFields = graphql(`
  fragment NightPlansFields on SNightTimelines {
    nightTimeline {
      nightIndex
      timeEntriesBySite {
        site
        mornTwilight
        eveTwilight
        nightStart
        nightEnd
        moonAltitude {
          time
          altitude
        }
        moonIllumination
        moonWaxing
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
              wv
              windSpeed
              windDirection
            }
            visits {
              obsId
              exploreUrl
              targets
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
              band
              groupId
              groupName
              completion
              atomTimes
              stepStart
              stepEnd
              peakScore
              requiredConditions {
                iq
                cc
                sb
                wv
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
`);

export const subscriptionQueueSchedule = graphql(`
  subscription queueSchedule($scheduleId: String!) {
    queueSchedule(scheduleId: $scheduleId) {
      __typename
      ... on NewNightPlans {
        nightPlans {
          ...NightPlansFields
        }
        plansSummary {
          summary
          metricsPerBand
        }
      }
      ... on ProposedNightPlans {
        proposalId
        nightPlans {
          ...NightPlansFields
        }
        plansSummary {
          summary
          metricsPerBand
        }
      }
      ... on PlanDecision {
        proposalId
        accepted
        decidedBy
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
              band
              groupId
              groupName
              obsId
              exploreUrl
              peakScore
              score
              startTime
              requiredConditions {
                cc
                iq
                sb
                wv
              }
            }
            nightConditions {
              cc
              iq
              wv
              windSpeed
              windDirection
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
              band
              groupId
              groupName
              obsId
              exploreUrl
              peakScore
              score
              startTime
              requiredConditions {
                cc
                iq
                sb
                wv
              }
            }
            nightConditions {
              cc
              iq
              wv
              windSpeed
              windDirection
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
