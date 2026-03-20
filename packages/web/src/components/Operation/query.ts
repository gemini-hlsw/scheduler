import { graphql } from "@/../../schema/web";
export const scheduleV2Query = graphql(`
  query scheduleV2(
    $startTime: String!
    $endTime: String!
    $nightStartTime: String!
    $nightEndTime: String!
    $sites: Sites!
    $imageQuality: Float!
    $cloudCover: Float!
    $windSpeed: Float!
    $windDirection: Float!
    $thesisFactor: Float
    $power: Int
    $metPower: Float
    $whaPower: Float
    $airPower: Float
    $visPower: Float
    $programs: [String!]!
  ) {
    scheduleV2(
      newScheduleRtInput: {
        startTime: $startTime
        endTime: $endTime
        nightStartTime: $nightStartTime
        nightEndTime: $nightEndTime
        sites: $sites
        imageQuality: $imageQuality
        cloudCover: $cloudCover
        windSpeed: $windSpeed
        windDirection: $windDirection
        thesisFactor: $thesisFactor
        power: $power
        metPower: $metPower
        whaPower: $whaPower
        airPower: $airPower
        visPower: $visPower
        programs: $programs
      }
    )
  }
`);
