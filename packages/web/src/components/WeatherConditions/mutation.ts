import { graphql } from "@/../../schema/web";
export const updateWeatherMutation = graphql(`
  mutation updateWeather($weatherInput: WeatherInput) {
    updateWeather(weatherInput: $weatherInput) {
      site
      imageQuality
      cloudCover
      windDirection
      windSpeed
    }
  }
`);
