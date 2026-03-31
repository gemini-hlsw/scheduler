import { graphql } from "@/gql";
export const scheduleQuery = graphql(`
  query schedule(
    $scheduleId: String!
    $startTime: String!
    $endTime: String!
    $sites: Sites!
    $mode: SchedulerModes!
    $numNightsToSchedule: Int!
    $semesterVisibility: Boolean!
    $thesisFactor: Float
    $power: Int
    $metPower: Float
    $whaPower: Float
    $airPower: Float
    $visPower: Float
    $programs: [String!]!
  ) {
    schedule(
      scheduleId: $scheduleId
      newScheduleInput: {
        startTime: $startTime
        sites: $sites
        mode: $mode
        endTime: $endTime
        thesisFactor: $thesisFactor
        power: $power
        metPower: $metPower
        whaPower: $whaPower
        airPower: $airPower
        visPower: $visPower
        semesterVisibility: $semesterVisibility
        numNightsToSchedule: $numNightsToSchedule
        programs: $programs
      }
    )
  }
`);
