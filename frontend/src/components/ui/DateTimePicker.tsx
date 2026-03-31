import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Calendar } from "@/components/ui/calendar";
import { useState } from "react";
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

  return (
    <div className="flex flex-row">
      <Popover open={isOpen} onOpenChange={setIsOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            id="date-picker-range"
            className={cn(
              "justify-start px-2.5 font-normal",
              "border-r-0 rounded-br-none rounded-tr-none"
            )}
          >
            {date ? format(date, "MM/dd/yyyy") : "Select date"}
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
        type="time"
        id="time-end"
        step="60"
        max="23:59"
        min="00:00"
        value={date ? format(date, "HH:mm") : ""}
        onChange={(e) => {
          const timeParts = e.target.value.split(":");
          if (timeParts.length === 2) {
            const [hours, minutes] = timeParts.map(Number);
            const newDate = new Date(date);
            newDate.setHours(hours);
            newDate.setMinutes(minutes);
            newDate.setSeconds(0);
            setDate(newDate);
          }
        }}
        className={cn(
          "bg-background appearance-none",
          "[&::-webkit-calendar-picker-indicator]:hidden",
          "[&::-webkit-calendar-picker-indicator]:appearance-none",
          "rounded-tl-none rounded-bl-none"
        )}
      />
    </div>
  );
}
