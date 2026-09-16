import { useContext } from "react";
import { GlobalStateContext } from "../GlobalState/GlobalState";
import { Input } from "@/components/ui/input";
import { Field, FieldLabel } from "@/components/ui/field";
import { cn } from "@/lib/utils";
import { RankerName } from "@/gql/graphql";

export default function RankerTweaker({
  vertical = false,
}: {
  vertical?: boolean;
}) {
  const {
    thesis,
    setThesis,
    power,
    setPower,
    metPower,
    setMetPower,
    visPower,
    setVisPower,
    whaPower,
    setWhaPower,
    airPower,
    setAirPower,
    ranker,
    setRanker,
  } = useContext(GlobalStateContext);

  // The Default ranker reads met/vis/wha as exponents, the Additive one as linear weights,
  // so the same number scores differently. Say which, next to the fields it applies to.
  const powerMeaning =
    ranker === "ADDITIVE"
      ? "Additive: MET/VIS/WHA are linear weights (terms are summed)."
      : "Default: MET/VIS/WHA are exponents (terms are multiplied).";

  return (
    <div
      className={cn(
        "border rounded-md flex flex-col gap-2 p-3 flex-wrap",
        "dark:bg-white/20 bg-black/10"
      )}
    >
      <h1 className="font-bold">Ranker Tweaker</h1>
      <div
        className={cn(
          "flex gap-1 items-center",
          // Wrap rather than squeeze: at phone width the row used to crush every field,
          // and the ranker select collapsed to just its dropdown arrow.
          vertical ? "flex-col" : "flex-row flex-wrap"
        )}
      >
        <Field orientation={vertical ? "horizontal" : "vertical"}>
          <FieldLabel className={vertical ? "w-1/2" : "w-fit"} htmlFor="ranker">
            Ranker
          </FieldLabel>
          <select
            className={cn(
              "border-input dark:bg-input/30 h-9 min-w-0 rounded-md border bg-transparent",
              "px-3 py-1 text-base shadow-xs outline-none md:text-sm",
              "focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]",
              vertical ? "w-1/2" : "w-fit"
            )}
            id="ranker"
            title={powerMeaning}
            onChange={(e) => setRanker(e.target.value as RankerName)}
            value={ranker}
          >
            {/* Field forces [&>*]:w-full, so long labels get clipped. The multiply/sum
                distinction is spelled out in the hint line below instead. */}
            <option value="DEFAULT">Default</option>
            <option value="ADDITIVE">Additive</option>
          </select>
        </Field>
        <Field orientation={vertical ? "horizontal" : "vertical"}>
          <FieldLabel className={vertical ? "w-1/2" : "w-fit"} htmlFor="thesis">
            Thesis factor
          </FieldLabel>
          <Input
            className={vertical ? "w-1/2" : "w-fit"}
            id="thesis"
            onChange={(e) => setThesis(parseFloat(e.target.value) || 0)}
            value={thesis}
            type="number"
            step={0.00001}
            aria-invalid={thesis < 0 || thesis > 10}
          />
        </Field>
        <Field orientation={vertical ? "horizontal" : "vertical"}>
          <FieldLabel className={vertical ? "w-1/2" : "w-fit"} htmlFor="power">
            Power factor
          </FieldLabel>
          <Input
            className={vertical ? "w-1/2" : "w-fit"}
            id="power"
            onChange={(e) => setPower(parseFloat(e.target.value) || 0)}
            value={power}
            aria-invalid={!Number.isInteger(power)}
            type="number"
          />
        </Field>
        <Field orientation={vertical ? "horizontal" : "vertical"}>
          <FieldLabel
            className={vertical ? "w-1/2" : "w-fit"}
            htmlFor="metPower"
          >
            MET power
          </FieldLabel>
          <Input
            className={vertical ? "w-1/2" : "w-fit"}
            id="metPower"
            onChange={(e) => setMetPower(parseFloat(e.target.value) || 0)}
            value={metPower}
            type="number"
            step={0.00001}
            aria-invalid={metPower < 0 || metPower > 10}
          />
        </Field>
        <Field orientation={vertical ? "horizontal" : "vertical"}>
          <FieldLabel
            className={vertical ? "w-1/2" : "w-fit"}
            htmlFor="visPower"
          >
            Visibility power
          </FieldLabel>
          <Input
            className={vertical ? "w-1/2" : "w-fit"}
            id="visPower"
            onChange={(e) => setVisPower(parseFloat(e.target.value) || 0)}
            value={visPower}
            type="number"
            step={0.00001}
            aria-invalid={visPower < 0 || visPower > 10}
          />
        </Field>
        <Field orientation={vertical ? "horizontal" : "vertical"}>
          <FieldLabel
            className={vertical ? "w-1/2" : "w-fit"}
            htmlFor="whaPower"
          >
            WHA power
          </FieldLabel>
          <Input
            className={vertical ? "w-1/2" : "w-fit"}
            id="whaPower"
            onChange={(e) => setWhaPower(parseFloat(e.target.value) || 0)}
            value={whaPower}
            type="number"
            step={0.00001}
            aria-invalid={whaPower < 0 || whaPower > 10}
          />
        </Field>
        <Field orientation={vertical ? "horizontal" : "vertical"}>
          <FieldLabel
            className={vertical ? "w-1/2" : "w-fit"}
            htmlFor="airPower"
          >
            Air power
          </FieldLabel>
          <Input
            className={vertical ? "w-1/2" : "w-fit"}
            id="airPower"
            onChange={(e) => setAirPower(parseFloat(e.target.value) || 0)}
            value={airPower}
            type="number"
            step={0.00001}
            aria-invalid={airPower < 0 || airPower > 10}
          />
        </Field>
      </div>
      <p className="text-muted-foreground text-xs">{powerMeaning}</p>
    </div>
  );
}
