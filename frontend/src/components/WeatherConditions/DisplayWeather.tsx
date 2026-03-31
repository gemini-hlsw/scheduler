import { useQuery, useReactiveVar, useSubscription } from "@apollo/client";
import { weatherUpdatesSubscription } from "./subscription";
import { cn } from "@/lib/utils";
import { getWeather } from "./query";
import { useEffect, useState } from "react";
import { Table, TableBody, TableCell, TableRow } from "../ui/table";
import { isWeatherConnectedVar } from "@/apollo-client";
import { ConnectionBadge } from "../ui/connectionBadge";

export function DisplayWeather() {
  const isOnline = useReactiveVar(isWeatherConnectedVar);

  const { data: weatherData } = useQuery(getWeather, {
    fetchPolicy: "no-cache",
    context: { clientName: "weatherClient" },
  });

  const { data, error } = useSubscription(weatherUpdatesSubscription, {
    context: { clientName: "weatherClient" },
  });

  const [weatherState, setWeatherState] = useState([]);

  useEffect(() => {
    setWeatherState(weatherData?.weather);
  }, [weatherData]);

  useEffect(() => {
    if (data && data.weatherUpdates) {
      const currentState = [...weatherState];
      setWeatherState(
        currentState?.map((w) => {
          if (w.site === data.weatherUpdates.site) {
            return data.weatherUpdates;
          } else {
            return w;
          }
        })
      );
    }
  }, [data]);

  return (
    <div
      className={cn(
        "border rounded-md flex flex-col gap-2 p-3 flex-wrap grow",
        "dark:bg-white/20 bg-black/10"
      )}
    >
      <h1 className="font-bold flex justify-between">
        <span>Current Weather</span>
        <ConnectionBadge isOnline={isOnline} />
      </h1>
      {error && <p>Error loading weather data: {error.message}</p>}
      <Table>
        <TableBody>
          <TableRow
            className={cn(
              "odd:bg-muted/50 *:px-2",
              "dark:hover:bg-white/30 hover:bg-black/30"
            )}
          >
            <TableCell>Site</TableCell>
            {weatherState?.map((data: { site: string }) => (
              <TableCell key={`site${data.site}`}>{data.site}</TableCell>
            ))}
          </TableRow>
          <TableRow
            className={cn(
              "odd:bg-muted/50 *:px-2",
              "dark:hover:bg-white/30 hover:bg-black/30"
            )}
          >
            <TableCell>Image Quality</TableCell>
            {weatherState?.map(
              (data: { site: string; imageQuality: number }) => (
                <TableCell key={`imageQuality${data.site}`}>
                  {data.imageQuality}
                </TableCell>
              )
            )}
          </TableRow>
          <TableRow
            className={cn(
              "odd:bg-muted/50 *:px-2",
              "dark:hover:bg-white/30 hover:bg-black/30"
            )}
          >
            <TableCell>Cloud Cover</TableCell>
            {weatherState?.map((data: { site: string; cloudCover: number }) => (
              <TableCell key={`cloudCover${data.site}`}>
                {data.cloudCover}
              </TableCell>
            ))}
          </TableRow>
          <TableRow
            className={cn(
              "odd:bg-muted/50 *:px-2",
              "dark:hover:bg-white/30 hover:bg-black/30"
            )}
          >
            <TableCell>Wind Direction</TableCell>
            {weatherState?.map(
              (data: { site: string; windDirection: number }) => (
                <TableCell key={`windDirection${data.site}`}>
                  {data.windDirection}
                </TableCell>
              )
            )}
          </TableRow>
          <TableRow
            className={cn(
              "odd:bg-muted/50 *:px-2",
              "dark:hover:bg-white/30 hover:bg-black/30"
            )}
          >
            <TableCell>Wind Speed</TableCell>
            {weatherState?.map((data: { site: string; windSpeed: number }) => (
              <TableCell key={`windSpeed${data.site}`}>
                {data.windSpeed}
              </TableCell>
            ))}
          </TableRow>
        </TableBody>
      </Table>
    </div>
  );
}
