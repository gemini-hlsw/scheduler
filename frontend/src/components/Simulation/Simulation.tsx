import WeatherConditions from "../WeatherConditions/WeatherConditions";
import { DisplayWeather } from "../WeatherConditions/DisplayWeather";

export function Simulation() {
  return (
    <div className="flex flex-col gap-2">
      <WeatherConditions updateButton={true} />
      <DisplayWeather />
    </div>
  );
}
