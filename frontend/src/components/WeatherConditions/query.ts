import { graphql } from "@/gql";
export const getWeather = graphql(`
  query getWeather {
    weather {
      site
      imageQuality
      cloudCover
      waterVapor
      skyBackground
      windDirection
      windSpeed
    }
  }
`);
