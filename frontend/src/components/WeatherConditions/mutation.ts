import { graphql } from "@/gql";
export const updateWeatherMutation = graphql(`
  mutation updateWeather($weatherInput: WeatherInput) {
    updateWeather(weatherInput: $weatherInput) {
      site
      imageQuality
      cloudCover
      waterVapor
      windDirection
      windSpeed
    }
  }
`);
