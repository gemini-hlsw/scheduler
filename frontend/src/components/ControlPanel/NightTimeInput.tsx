import { useEffect, useState } from "react";
import { format } from "date-fns";
import { toast } from "sonner";
import { Label } from "@/components/ui/label";
import { Field } from "@/components/ui/field";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

// The times a night start or end can take, in site wall-clock time (see
// toSiteWallClock). Both ends are accepted.
export interface NightWindow {
  from: Date;
  to: Date;
}

// The time of day `text` (HH:mm) names inside `window`, or undefined when it
// is not in it. A night is shorter than a day, so at most one of the two
// calendar dates it spans can hold that time.
export function resolveNightTime(
  text: string,
  window: NightWindow,
): Date | undefined {
  const match = text.trim().match(/^([01]?\d|2[0-3]):?([0-5]\d)$/);
  if (!match) return undefined;
  const [, hours, minutes] = match;
  for (const dayOffset of [0, 1]) {
    const candidate = new Date(window.from);
    candidate.setDate(candidate.getDate() + dayOffset);
    candidate.setHours(Number(hours), Number(minutes), 0, 0);
    if (window.from <= candidate && candidate <= window.to) return candidate;
  }
  return undefined;
}

// A time-only input for a night start or end: the date follows from the
// night window, so 22:00 lands on the evening and 03:00 on the next morning.
export function NightTimeInput({
  title,
  dateTime,
  setDateTime,
  window,
  setToNow,
  vertical = false,
  clearButton = null,
}: {
  title: string;
  dateTime: Date | undefined;
  setDateTime: (date: Date | undefined) => void;
  window: NightWindow | undefined;
  setToNow?: () => void;
  vertical?: boolean;
  clearButton?: React.ReactNode;
}) {
  const [timeText, setTimeText] = useState("");

  useEffect(() => {
    setTimeText(dateTime ? format(dateTime, "HH:mm") : "");
  }, [dateTime]);

  const range = window
    ? `${format(window.from, "yyyy-MM-dd HH:mm")} to ${format(window.to, "yyyy-MM-dd HH:mm")}`
    : "";

  const commitTime = (value: string) => {
    if (value.trim() === "") {
      setDateTime(undefined);
      return;
    }
    const resolved = window ? resolveNightTime(value, window) : undefined;
    if (resolved) {
      setDateTime(resolved);
      setTimeText(format(resolved, "HH:mm"));
    } else {
      // Back to the time that is actually set, so the field never shows one
      // that would not be sent.
      setTimeText(dateTime ? format(dateTime, "HH:mm") : "");
      toast.error(
        window
          ? `${title}: "${value}" is not a time between ${range}.`
          : `${title}: the night is not known yet.`,
      );
    }
  };

  return (
    <Field orientation="horizontal" className="">
      <Label
        className={cn("text-nowrap", vertical ? "w-36 shrink-0" : "w-fit")}
      >
        {title}
      </Label>
      <div
        className="flex flex-row"
        title={window ? `Between ${range}` : "Night not known yet"}
      >
        <div
          className={cn(
            "flex items-center px-2.5 w-32 h-9 text-sm",
            "border rounded-md border-r-0 rounded-br-none rounded-tr-none",
            "text-muted-foreground",
          )}
        >
          {dateTime ? format(dateTime, "yyyy-MM-dd") : "Tonight"}
        </div>
        <Input
          type="text"
          inputMode="numeric"
          placeholder="HH:mm"
          disabled={!window}
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
      {setToNow && (
        <Button
          className={cn(
            "dark:bg-blue-800 bg-blue-400",
            "dark:text-white text-black text-sm",
            "dark:hover:bg-blue-700 hover:bg-blue-500",
            // Same width as the visibility nights count above it.
            "w-16",
          )}
          disabled={!window}
          onClick={setToNow}
        >
          Now
        </Button>
      )}
      {clearButton}
    </Field>
  );
}
