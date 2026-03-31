import { Field, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

export function NightsNumber({
  vertical,
  nights,
  setNights,
}: {
  vertical?: boolean;
  nights: number;
  setNights: (nights: number) => void;
}) {
  return (
    <Field orientation={"horizontal"} className="w-fit">
      <FieldLabel
        className={cn("text-nowrap", vertical ? "w-32" : "w-fit")}
        htmlFor="nights"
      >
        Nights
      </FieldLabel>
      <Input
        className={cn(vertical ? "w-1/2" : "w-20")}
        id="nights"
        onChange={(e) => setNights(parseFloat(e.target.value) || 0)}
        value={nights}
        type="number"
        step={1}
        aria-invalid={nights < 0 || nights > 183}
      />
    </Field>
  );
}
