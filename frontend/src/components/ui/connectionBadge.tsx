import { cn } from "@/lib/utils";
import { Badge } from "./badge";

export function ConnectionBadge({
  isOnline,
  text = "",
  className,
}: {
  isOnline: boolean;
  text?: string;
  className?: string;
}) {
  return (
    <Badge
      className={cn(
        "text-xs text-white",
        isOnline ? "bg-green-600" : "bg-red-600",
        className
      )}
      variant="default"
    >
      {text}
      {isOnline ? "Online" : "Offline"}
    </Badge>
  );
}
