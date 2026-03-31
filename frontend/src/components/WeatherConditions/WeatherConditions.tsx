import { useContext, useState } from "react";
import { GlobalStateContext } from "../GlobalState/GlobalState";
import { Button } from "@/components/ui/button";
import { updateWeatherMutation } from "./mutation";
import { useMutation } from "@apollo/client";
import { Field, FieldLabel } from "@/components/ui/field";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { FaCloudSun } from "react-icons/fa";

interface OptionType {
  label: string;
  value: number;
}

const IQ_OPTIONS: OptionType[] = [
  { label: "IQ20", value: 0.2 },
  { label: "IQ70", value: 0.7 },
  { label: "IQ85", value: 0.85 },
  { label: "IQANY", value: 1.0 },
];

const CC_OPTIONS = [
  { label: "CC50", value: 0.5 },
  { label: "CC70", value: 0.7 },
  { label: "CC80", value: 0.8 },
  { label: "CCANY", value: 1.0 },
];

export default function WeatherConditions({
  vertical = false,
  updateButton = false,
}: {
  vertical?: boolean;
  updateButton?: boolean;
}) {
  const {
    imageQuality,
    setImageQuality,
    cloudCover,
    setCloudCover,
    windDirection,
    setWindDirection,
    windSpeed,
    setWindSpeed,
  } = useContext(GlobalStateContext);

  const [siteState, setSite] = useState(undefined);
  const sites = [
    { label: "GN", value: "GN" },
    { label: "GS", value: "GS" },
  ];

  const [updateWeather] = useMutation(updateWeatherMutation, {
    context: { clientName: "weatherClient" },
  });

  function sendWeatherUpdate() {
    updateWeather({
      variables: {
        weatherInput: {
          imageQuality,
          cloudCover,
          windDirection,
          windSpeed,
          site: siteState,
        },
      },
    });
  }

  return (
    <div
      className={cn(
        "border rounded-md flex flex-col gap-2 p-3 flex-wrap",
        "dark:bg-white/20 bg-black/10"
      )}
    >
      <h1 className="font-bold">Weather Conditions</h1>
      <div
        className={cn(
          "flex gap-1 items-center",
          vertical ? "flex-col" : "flex-row"
        )}
      >
        <Field orientation={vertical ? "horizontal" : "vertical"}>
          <FieldLabel
            className={vertical ? "w-1/2" : "w-fit"}
            htmlFor="image-quality"
          >
            Image quality
          </FieldLabel>
          <Combobox
            items={IQ_OPTIONS}
            itemToStringValue={(item: OptionType) => item.label}
            onValueChange={(e) => setImageQuality(e.value)}
          >
            <ComboboxInput
              className={vertical ? "w-1/2" : "w-fit"}
              placeholder="Select IQ"
              value={
                imageQuality
                  ? IQ_OPTIONS.find((i) => i.value === imageQuality)?.label
                  : undefined
              }
            />
            <ComboboxContent>
              <ComboboxEmpty>No items found.</ComboboxEmpty>
              <ComboboxList>
                {(item) => (
                  <ComboboxItem key={item.value} value={item}>
                    {item.label}
                  </ComboboxItem>
                )}
              </ComboboxList>
            </ComboboxContent>
          </Combobox>
        </Field>
        <Field orientation={vertical ? "horizontal" : "vertical"}>
          <FieldLabel
            className={vertical ? "w-1/2" : "w-fit"}
            htmlFor="cloud-cover"
          >
            Cloud cover
          </FieldLabel>
          <Combobox
            items={CC_OPTIONS}
            itemToStringValue={(item: OptionType) => item.label}
            onValueChange={(e) => setCloudCover(e.value)}
          >
            <ComboboxInput
              className={vertical ? "w-1/2" : "w-fit"}
              placeholder="Select CC"
              value={
                cloudCover
                  ? CC_OPTIONS.find((i) => i.value === cloudCover)?.label
                  : undefined
              }
            />
            <ComboboxContent>
              <ComboboxEmpty>No items found.</ComboboxEmpty>
              <ComboboxList>
                {(item) => (
                  <ComboboxItem key={item.value} value={item}>
                    {item.label}
                  </ComboboxItem>
                )}
              </ComboboxList>
            </ComboboxContent>
          </Combobox>
        </Field>
        <Field orientation={vertical ? "horizontal" : "vertical"}>
          <FieldLabel
            className={vertical ? "w-1/2" : "w-fit"}
            htmlFor="wind-dir"
          >
            Wind direction (deg)
          </FieldLabel>
          <Input
            className={vertical ? "w-1/2" : "w-fit"}
            id="wind-dir"
            onChange={(e) => setWindDirection(parseFloat(e.target.value) || 0)}
            value={windDirection}
            type="number"
            step={0.1}
            aria-invalid={windDirection < 0 || windDirection > 360}
          />
        </Field>
        <Field orientation={vertical ? "horizontal" : "vertical"}>
          <FieldLabel
            className={vertical ? "w-1/2" : "w-fit"}
            htmlFor="wind-speed"
          >
            Wind speed (m/s)
          </FieldLabel>
          <Input
            className={vertical ? "w-1/2" : "w-fit"}
            id="wind-speed"
            onChange={(e) => setWindSpeed(parseFloat(e.target.value) || 0)}
            value={windSpeed}
            type="number"
            step={0.1}
            aria-invalid={windSpeed < 0 || windSpeed > 100}
          />
        </Field>
        {updateButton && (
          <Field orientation={vertical ? "horizontal" : "vertical"}>
            <FieldLabel className={vertical ? "w-1/2" : "w-fit"} htmlFor="site">
              Site
            </FieldLabel>
            <Combobox
              items={sites}
              itemToStringValue={(item: OptionType) => item.label}
              onValueChange={(e) => setSite(e.value)}
            >
              <ComboboxInput
                className={vertical ? "w-1/2" : "w-fit"}
                placeholder="Select Site"
                value={
                  siteState
                    ? sites.find((i) => i.value === siteState)?.label
                    : ""
                }
              />
              <ComboboxContent>
                <ComboboxEmpty>No items found.</ComboboxEmpty>
                <ComboboxList>
                  {(item) => (
                    <ComboboxItem key={item.value} value={item}>
                      {item.label}
                    </ComboboxItem>
                  )}
                </ComboboxList>
              </ComboboxContent>
            </Combobox>
          </Field>
        )}
        {updateButton && (
          <Button
            className={cn(
              vertical ? "w-full" : "w-fit mt-8",
              "dark:text-white text-black dark:bg-green-800 bg-green-400",
              "dark:hover:bg-green-700 hover:bg-green-500"
            )}
            disabled={!siteState}
            onClick={sendWeatherUpdate}
          >
            <FaCloudSun />
            Send weather update
          </Button>
        )}
      </div>
    </div>
  );
}
