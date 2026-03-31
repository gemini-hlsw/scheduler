import { Label } from "@/components/ui/label";
import { Field } from "@/components/ui/field";
import { Button } from "@/components/ui/button";
import { DateTimePicker } from "@/components/ui/DateTimePicker";
import { cn } from "@/lib/utils";

export function DateTimeSelector({
  dateTime,
  setDateTime,
  setToNow,
  title = "UT Plan Start",
  setToNowButton = false,
  vertical = false,
  clearButton = null,
}: {
  dateTime: Date;
  setDateTime: (date: Date) => void;
  setToNow?: () => void;
  title?: string;
  setToNowButton?: boolean;
  vertical?: boolean;
  clearButton?: React.ReactNode;
}) {
  return (
    <Field orientation="horizontal" className="w-fit">
      <Label
        htmlFor="night-start"
        className={cn("text-nowrap", vertical ? "w-32" : "w-fit")}
      >
        {title}
      </Label>
      <DateTimePicker date={dateTime} setDate={setDateTime} />
      {setToNowButton && (
        <Button
          className={cn(
            "dark:bg-blue-800 bg-blue-400",
            "dark:text-white text-black",
            "dark:hover:bg-blue-700 hover:bg-blue-500"
          )}
          onClick={setToNow}
        >
          Set plan start to now
        </Button>
      )}
      {clearButton}
    </Field>
  );
}
