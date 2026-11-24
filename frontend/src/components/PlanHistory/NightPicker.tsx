import { useState } from "react";
import { CalendarIcon } from "lucide-react";
import { format } from "date-fns/format";
import { Button } from "@/components/ui/button";
import { Calendar } from "@/components/ui/calendar";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";

/** "2026-11-18" as that calendar day, not as UTC midnight shifted into the browser's zone. */
function toDay(isoDate: string): Date {
  const [year, month, day] = isoDate.split("-").map(Number);
  return new Date(year, month - 1, day);
}

/**
 * The night to show, picked like the Validation control panel's UT Visibility
 * range: a button that opens a calendar. Days with stored plans are marked.
 */
export function NightPicker({
  nightDate,
  onChange,
  nightsWithPlans,
}: {
  nightDate: string | null;
  onChange: (nightDate: string) => void;
  nightsWithPlans: string[];
}) {
  const [open, setOpen] = useState(false);
  const selected = nightDate ? toDay(nightDate) : undefined;

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          id="plan-history-night"
          className="justify-start px-2.5 font-normal w-44"
        >
          <CalendarIcon />
          {nightDate ?? <span>Pick a night</span>}
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-auto p-0" align="start">
        <Calendar
          mode="single"
          defaultMonth={selected}
          selected={selected}
          onSelect={(day) => {
            if (day) {
              onChange(format(day, "yyyy-MM-dd"));
              setOpen(false);
            }
          }}
          modifiers={{ withPlans: nightsWithPlans.map(toDay) }}
          modifiersClassNames={{
            withPlans:
              "font-bold underline decoration-fuchsia-500 decoration-2 underline-offset-4",
          }}
        />
        <p className="px-3 pb-2 text-xs text-muted-foreground">
          Underlined days have stored plans.
        </p>
      </PopoverContent>
    </Popover>
  );
}
