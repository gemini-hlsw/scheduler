import { cn } from "@/lib/utils";
import { type TimeEntryType } from "@/types";
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
} from "@/components/ui/tooltip";

export function TimelineBullets({
  date,
  timeline,
  selectedEntry,
  setSelectedEntry,
}: {
  date: string;
  timeline: TimeEntryType[];
  selectedEntry?: TimeEntryType;
  setSelectedEntry: (entry: TimeEntryType) => void;
}) {
  if (!timeline || timeline.length === 0) return null;

  const w =
    timeline.length > 1
      ? timeline.at(-1).startTimeSlots - timeline.at(0).startTimeSlots
      : 1;

  const startSlot = timeline.at(0).startTimeSlots;

  return (
    <div className="h-5 relative group">
      <div
        className={cn(
          "border absolute top-2 w-[calc(100%-24px)]",
          "group-hover:border-blue-500 dark:group-hover:border-blue-500",
          "transition-colors"
        )}
      >
        {timeline.map((en) => {
          const pos = ((en.startTimeSlots - startSlot) / w) * 100;
          return (
            <Tooltip key={`${en.event}${date}${en.startTimeSlots}`}>
              <TooltipTrigger
                key={en.startTimeSlots}
                onClick={() => setSelectedEntry(en)}
                className={cn(
                  "absolute -top-3 w-6 h-6 rounded-full",
                  "border border-black/20 dark:border-white/20",
                  JSON.stringify(en) === JSON.stringify(selectedEntry)
                    ? "bg-blue-500 dark:bg-blue-500"
                    : "bg-gray-400 dark:bg-gray-600",
                  "hover:bg-blue-500 dark:hover:bg-blue-500",
                  "transition-colors cursor-pointer"
                )}
                style={{ left: `${pos}%` }}
              />
              <TooltipContent>
                <p>{en.event}</p>
              </TooltipContent>
            </Tooltip>
          );
        })}
      </div>
    </div>
  );
}
