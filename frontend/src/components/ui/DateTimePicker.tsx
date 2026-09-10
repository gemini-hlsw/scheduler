import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Calendar } from "@/components/ui/calendar";
import { useEffect, useState } from "react";
import { format } from "date-fns";
import { ChevronDownIcon } from "lucide-react";
import { cn } from "@/lib/utils";

export function DateTimePicker({
  date,
  setDate,
}: {
  date: Date;
  setDate: (date: Date) => void;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [timeText, setTimeText] = useState(date ? format(date, "HH:mm") : "");

  useEffect(() => {
    setTimeText(date ? format(date, "HH:mm") : "");
  }, [date]);

  const commitTime = (value: string) => {
    const match = value.match(/^([01]?\d|2[0-3]):([0-5]?\d)$/);
    if (match) {
      const [, hoursStr, minutesStr] = match;
      const newDate = new Date(date);
      newDate.setHours(Number(hoursStr));
      newDate.setMinutes(Number(minutesStr));
      newDate.setSeconds(0);
      setDate(newDate);
    } else {
      setTimeText(date ? format(date, "HH:mm") : "");
    }
  };

  return (
    <div className="flex flex-row">
      <Popover open={isOpen} onOpenChange={setIsOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            id="date-picker-range"
            className={cn(
              "justify-start px-2.5 font-normal",
              "border-r-0 rounded-br-none rounded-tr-none",
              "w-32",
            )}
          >
            {date ? format(date, "yyyy-MM-dd") : "Select date"}
            <ChevronDownIcon />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-auto p-0" align="start">
          <Calendar
            mode="single"
            defaultMonth={date}
            selected={date}
            onSelect={(selectedDate) => {
              const newDate = new Date(date);
              newDate.setFullYear(selectedDate.getFullYear());
              newDate.setMonth(selectedDate.getMonth());
              newDate.setDate(selectedDate.getDate());
              setDate(newDate);
              setIsOpen(false);
            }}
            numberOfMonths={2}
          />
        </PopoverContent>
      </Popover>
      <Input
        type="text"
        inputMode="numeric"
        placeholder="00:00"
        value={timeText}
        onChange={(e) => setTimeText(e.target.value)}
        onBlur={(e) => commitTime(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            commitTime(e.currentTarget.value);
            e.currentTarget.blur();
          }
        }}
        className={cn(
          "bg-background",
          "rounded-tl-none rounded-bl-none",
          "w-18",
        )}
      />
    </div>
  );
}
