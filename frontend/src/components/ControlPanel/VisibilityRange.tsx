import { Label } from "@/components/ui/label";
import { Field } from "@/components/ui/field";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Calendar } from "@/components/ui/calendar";
import { CalendarIcon } from "lucide-react";
import { format } from "date-fns/format";
import { cn } from "@/lib/utils";
import { DateRange } from "react-day-picker";

export function VisibilityRange({
  date,
  setDate,
  vertical = false,
  clearButton = null,
}: {
  date?: DateRange;
  setDate?: (date: DateRange) => void;
  vertical?: boolean;
  clearButton?: React.ReactNode;
}) {
  return (
    <Field orientation="horizontal" className={cn(vertical ? "" : "w-fit")}>
      <Label
        htmlFor="range"
        className={cn("text-nowrap", vertical ? "w-32" : "w-fit")}
      >
        UT Visibility
      </Label>
      <Popover>
        <PopoverTrigger asChild className="grow">
          <Button
            variant="outline"
            id="date-picker-range"
            className="justify-start px-2.5 font-normal"
          >
            <CalendarIcon />
            {date?.from ? (
              date.to ? (
                <>
                  {format(date.from, "MM/dd/yyyy")} -{" "}
                  {format(date.to, "MM/dd/yyyy")}
                </>
              ) : (
                format(date.from, "MM/dd/yyyy")
              )
            ) : (
              <span>Pick a date</span>
            )}
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-auto p-0" align="start">
          <Calendar
            mode="range"
            defaultMonth={date?.from}
            selected={date}
            onSelect={setDate}
            numberOfMonths={2}
          />
        </PopoverContent>
      </Popover>
      {clearButton}
    </Field>
  );
}
