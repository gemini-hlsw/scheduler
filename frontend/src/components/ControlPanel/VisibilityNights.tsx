import { useEffect, useState } from "react";
import { Label } from "@/components/ui/label";
import { Field } from "@/components/ui/field";
import { Button } from "@/components/ui/button";
import { ButtonGroup } from "@/components/ui/button-group";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

// How far past tonight an operation build computes visibility: to the end of
// the semester (nights undefined), or a number of nights.
export function VisibilityNights({
  nights,
  setNights,
  vertical = false,
}: {
  nights: number | undefined;
  setNights: (nights: number | undefined) => void;
  vertical?: boolean;
}) {
  const [text, setText] = useState(nights?.toString() ?? "");
  useEffect(() => setText(nights?.toString() ?? ""), [nights]);

  // Only whole numbers from 1 up are taken; anything else goes back to the
  // count that is set.
  const commit = (value: string) => {
    const count = Number(value);
    if (Number.isInteger(count) && count >= 1) setNights(count);
    else setText(nights?.toString() ?? "");
  };

  const buttonClass = (active: boolean) =>
    cn(
      "dark:text-white text-black",
      "dark:hover:bg-blue-700 hover:bg-blue-500",
      "px-2 flex-1",
      active ? "dark:bg-blue-800 bg-blue-400" : "dark:bg-white/20 bg-black/20"
    );

  return (
    <Field orientation="horizontal" className="">
      <Label
        className={cn("text-nowrap", vertical ? "w-36 shrink-0" : "w-fit")}
      >
        Visibility
      </Label>
      {/* The buttons take the width of the night inputs below (date box +
          time) and the count lines up with their Now buttons. */}
      <ButtonGroup className="w-50">
        <Button
          variant="default"
          className={buttonClass(nights === undefined)}
          title="From tonight to the end of the semester"
          onClick={() => setNights(undefined)}
        >
          Semester end
        </Button>
        <Button
          variant="default"
          className={buttonClass(nights !== undefined)}
          title="From tonight, for a number of nights"
          onClick={() => setNights(nights ?? 1)}
        >
          Nights
        </Button>
      </ButtonGroup>
      <Input
        type="number"
        min={1}
        step={1}
        aria-label="Visibility nights"
        placeholder="—"
        disabled={nights === undefined}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onBlur={(e) => commit(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") e.currentTarget.blur();
        }}
        className="bg-background w-16 px-2"
      />
    </Field>
  );
}
