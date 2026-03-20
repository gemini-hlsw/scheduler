import { graphql } from "@/../../schema/web";
export const getWeather = graphql(`
  query getWeather {
    weather {
      site
      imageQuality
      cloudCover
      windDirection
      windSpeed
    }
  }
`);
