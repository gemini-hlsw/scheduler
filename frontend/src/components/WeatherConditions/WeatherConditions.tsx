import { useContext, useEffect, useState } from "react";
import { GlobalStateContext } from "../GlobalState/GlobalState";
import { Button } from "@/components/ui/button";
import { updateWeatherMutation } from "./mutation";
import { useMutation } from "@apollo/client";
import {
  Field,
  FieldLabel,
  FieldLegend,
  FieldSet,
} from "@/components/ui/field";
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
import { iqToPercentile } from "./imageQuality";

interface OptionType<T = number> {
  label: string;
  value: T;
}

// Values are percentiles, as the weather service expects.
const CC_OPTIONS: OptionType[] = [
  { label: "0.00 mag", value: 0.5 },
  { label: "<0.1 mag", value: 0.55 },
  { label: "<0.3 mag", value: 0.7 },
  { label: "<0.5 mag", value: 0.75 },
  { label: "<1.0 mag", value: 0.85 },
  { label: "<2.0 mag", value: 0.95 },
  { label: "<3.0 mag", value: 1.0 },
];

const WV_OPTIONS: OptionType[] = [
  { label: "Very Dry", value: 0.2 },
  { label: "Dry", value: 0.5 },
  { label: "Median", value: 0.8 },
  { label: "Wet", value: 1.0 },
];

const SITE_OPTIONS: OptionType<string>[] = [
  { label: "GN", value: "GN" },
  { label: "GS", value: "GS" },
];

function OptionSelect<T>({
  options,
  value,
  onChange,
  placeholder,
  className,
  disabled = false,
}: {
  options: OptionType<T>[];
  value: T | undefined;
  onChange: (value: T) => void;
  placeholder: string;
  className?: string;
  disabled?: boolean;
}) {
  return (
    <Combobox
      disabled={disabled}
      items={options}
      itemToStringValue={(item: OptionType<T>) => item.label}
      onValueChange={(e) => e && onChange(e.value)}
    >
      <ComboboxInput
        className={className}
        disabled={disabled}
        placeholder={placeholder}
        value={options.find((i) => i.value === value)?.label ?? ""}
      />
      <ComboboxContent>
        <ComboboxEmpty>No items found.</ComboboxEmpty>
        <ComboboxList>
          {(item: OptionType<T>) => (
            <ComboboxItem key={item.label} value={item}>
              {item.label}
            </ComboboxItem>
          )}
        </ComboboxList>
      </ComboboxContent>
    </Combobox>
  );
}

/** A condition menu with the percentile of the selected option next to it. */
function PercentSelect({
  className,
  vertical,
  ...props
}: {
  vertical: boolean;
  options: OptionType[];
  value: number;
  onChange: (value: number) => void;
  placeholder: string;
  className?: string;
  disabled?: boolean;
}) {
  return (
    <div className={cn(className, "flex items-center gap-2")}>
      <OptionSelect {...props} className="flex-1 min-w-0" />
      <span
        className={cn(
          "text-sm tabular-nums shrink-0",
          vertical ? "w-12 text-right" : "w-10",
          props.disabled && "opacity-50",
        )}
      >
        {`${Math.round(props.value * 100)} %`}
      </span>
    </div>
  );
}

/**
 * A number input that keeps the text as typed, so clearing a field leaves it empty instead
 * of putting a 0 back (which then turned "5" into "05"). An empty or unparsable field
 * reports NaN, which the panel treats as invalid.
 */
function NumberInput({
  value,
  onValueChange,
  ...props
}: Omit<React.ComponentProps<typeof Input>, "value" | "onChange" | "type"> & {
  value: number;
  onValueChange: (value: number) => void;
}) {
  const [text, setText] = useState(Number.isNaN(value) ? "" : String(value));

  // Follow changes made elsewhere (the other tab's panel shares this state), but leave the
  // text alone while it already reads as the current value.
  useEffect(() => {
    if (!Object.is(parseFloat(text), value)) {
      setText(Number.isNaN(value) ? "" : String(value));
    }
  }, [value]);

  return (
    <Input
      {...props}
      type="number"
      value={text}
      onChange={(e) => {
        // Typing after an existing 0 gives "05": drop the leading zeros ("0.5" is kept).
        const typed = e.target.value.replace(/^(-?)0+(?=\d)/, "$1");
        setText(typed);
        onValueChange(parseFloat(typed));
      }}
    />
  );
}

export default function WeatherConditions({
  vertical = false,
  updateButton = false,
}: {
  vertical?: boolean;
  updateButton?: boolean;
}) {
  const {
    fwhm,
    setFwhm,
    airmass,
    setAirmass,
    wavelength,
    setWavelength,
    cloudCover,
    setCloudCover,
    waterVapor,
    setWaterVapor,
    windDirection,
    setWindDirection,
    windSpeed,
    setWindSpeed,
  } = useContext(GlobalStateContext);

  const [siteState, setSite] = useState<string | undefined>(undefined);

  // A FWHM of 0 gives 0 %, the same as any FWHM below ~0.37" at 0.55 µm.
  const fwhmValid = fwhm >= 0 && fwhm <= 5;
  const airmassValid = airmass >= 1 && airmass <= 3;
  const wavelengthValid = wavelength > 0;
  const iqInputsValid = fwhmValid && airmassValid && wavelengthValid;
  const windDirectionValid = windDirection >= 0 && windDirection <= 360;
  const windSpeedValid = windSpeed >= 0 && windSpeed <= 100;
  const iqPercentile = iqInputsValid
    ? iqToPercentile(fwhm, wavelength, airmass)
    : undefined;

  const [updateWeather] = useMutation(updateWeatherMutation, {
    context: { clientName: "weatherClient" },
  });

  function sendWeatherUpdate() {
    if (iqPercentile === undefined || !windDirectionValid || !windSpeedValid) {
      return;
    }
    updateWeather({
      variables: {
        weatherInput: {
          imageQuality: iqPercentile / 100,
          cloudCover,
          waterVapor,
          windDirection,
          windSpeed,
          site: siteState,
        },
      },
    });
  }

  // Vertical: two columns (image quality | other conditions) with the site and the
  // send button underneath, so the panel stays about as tall as Build Parameters.
  const labelClass = vertical ? "whitespace-nowrap" : "w-fit";
  const controlClass = vertical ? "w-44 shrink-0" : "w-fit";
  const orientation = vertical ? "horizontal" : "vertical";
  const iqLabelClass = vertical ? labelClass : "w-fit whitespace-nowrap";
  const iqInputClass = vertical ? "w-20 shrink-0" : "w-full";
  const iqFieldClass = vertical ? undefined : "w-28 shrink-0";

  return (
    <div
      className={cn(
        "border rounded-md flex flex-col gap-2 p-3 flex-wrap",
        "dark:bg-white/20 bg-black/10",
      )}
    >
      <h1 className="font-bold">Weather Conditions</h1>
      <div
        className={cn(
          vertical
            ? "grid grid-cols-[auto_auto] gap-x-4 gap-y-2 items-start"
            : "flex flex-row gap-4 items-center",
        )}
      >
        <FieldSet
          className={cn(
            "border rounded-md p-2 gap-1",
            vertical ? "w-full" : "w-fit shrink-0",
          )}
        >
          <FieldLegend variant="label" className="mb-0 px-1">
            Image quality
          </FieldLegend>
          <div
            className={cn(
              "flex gap-1",
              vertical ? "flex-col" : "flex-row items-end",
            )}
          >
            <Field orientation={orientation} className={iqFieldClass}>
              <FieldLabel className={iqLabelClass} htmlFor="iq-fwhm">
                FWHM (arcsec)
              </FieldLabel>
              <NumberInput
                className={iqInputClass}
                id="iq-fwhm"
                onValueChange={setFwhm}
                value={fwhm}
                step={0.01}
                min={0}
                max={5}
                aria-invalid={!fwhmValid}
              />
            </Field>
            <Field orientation={orientation} className={iqFieldClass}>
              <FieldLabel className={iqLabelClass} htmlFor="iq-airmass">
                Airmass
              </FieldLabel>
              <NumberInput
                className={iqInputClass}
                id="iq-airmass"
                onValueChange={setAirmass}
                value={airmass}
                step={0.01}
                min={1}
                max={3}
                aria-invalid={!airmassValid}
              />
            </Field>
            <Field orientation={orientation} className={iqFieldClass}>
              <FieldLabel className={iqLabelClass} htmlFor="iq-wavelength">
                Wavelength (µm)
              </FieldLabel>
              <NumberInput
                className={iqInputClass}
                id="iq-wavelength"
                onValueChange={setWavelength}
                value={wavelength}
                step={0.01}
                aria-invalid={!wavelengthValid}
              />
            </Field>
            <Field orientation={orientation} className={iqFieldClass}>
              <FieldLabel className={cn(iqLabelClass, "font-semibold")}>
                Percentile
              </FieldLabel>
              <span
                className={cn(
                  iqInputClass,
                  "text-sm font-semibold tabular-nums",
                  !vertical && "h-9 flex items-center",
                )}
              >
                {iqPercentile === undefined
                  ? "Invalid inputs"
                  : `${iqPercentile.toFixed(1)} %`}
              </span>
            </Field>
          </div>
        </FieldSet>
        <div className={vertical ? "flex flex-col gap-1" : "contents"}>
          <Field orientation={orientation}>
            <FieldLabel className={labelClass} htmlFor="cloud-cover">
              Cloud cover
            </FieldLabel>
            <PercentSelect
              options={CC_OPTIONS}
              value={cloudCover}
              onChange={setCloudCover}
              placeholder="Select CC"
              className={controlClass}
              vertical={vertical}
            />
          </Field>
          <Field orientation={orientation}>
            <FieldLabel className={labelClass} htmlFor="water-vapor">
              Water vapor
            </FieldLabel>
            <PercentSelect
              options={WV_OPTIONS}
              value={waterVapor}
              onChange={setWaterVapor}
              placeholder="Select WV"
              className={controlClass}
              vertical={vertical}
              // Not used by the scheduler yet: keep the default until it is.
              disabled
            />
          </Field>
          <Field orientation={orientation}>
            <FieldLabel className={labelClass} htmlFor="wind-dir">
              Wind direction (deg)
            </FieldLabel>
            <NumberInput
              className={controlClass}
              id="wind-dir"
              onValueChange={setWindDirection}
              value={windDirection}
              step={0.1}
              aria-invalid={!windDirectionValid}
            />
          </Field>
          <Field orientation={orientation}>
            <FieldLabel className={labelClass} htmlFor="wind-speed">
              Wind speed (m/s)
            </FieldLabel>
            <NumberInput
              className={controlClass}
              id="wind-speed"
              onValueChange={setWindSpeed}
              value={windSpeed}
              step={0.1}
              aria-invalid={!windSpeedValid}
            />
          </Field>
        </div>
        <div
          className={
            vertical
              ? "col-span-2 flex flex-row gap-4 items-center"
              : "contents"
          }
        >
          {updateButton && (
            <Field
              orientation={orientation}
              className={vertical ? "w-fit" : undefined}
            >
              <FieldLabel className={labelClass} htmlFor="site">
                Site
              </FieldLabel>
              <OptionSelect
                options={SITE_OPTIONS}
                value={siteState}
                onChange={setSite}
                placeholder="Select Site"
                className={controlClass}
              />
            </Field>
          )}
          {updateButton && (
            <Button
              className={cn(
                vertical ? "flex-1" : "w-fit mt-8",
                "dark:text-white text-black dark:bg-green-800 bg-green-400",
                "dark:hover:bg-green-700 hover:bg-green-500",
              )}
              disabled={
                !siteState ||
                iqPercentile === undefined ||
                !windDirectionValid ||
                !windSpeedValid
              }
              onClick={sendWeatherUpdate}
            >
              <FaCloudSun />
              Send weather update
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
