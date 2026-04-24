import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { ImSpinner9 } from "react-icons/im";
import { FaPlay } from "react-icons/fa";

export function RunButton({
  loadingPlan,
  run,
  isRunDisabled,
  title = "RUN",
  icon = <FaPlay />,
  full = false,
}: {
  loadingPlan: boolean;
  run: () => void;
  isRunDisabled: boolean;
  title?: string;
  icon?: React.ReactNode;
  full?: boolean;
}) {
  return (
    <Button
      variant="default"
      className={cn(
        "dark:text-white text-black dark:bg-green-800 bg-green-400",
        "dark:hover:bg-green-700 hover:bg-green-500",
        full ? "w-full" : ""
      )}
      disabled={isRunDisabled || loadingPlan}
      onClick={run}
    >
      {loadingPlan ? <ImSpinner9 className="animate-spin" /> : icon}
      {title}
    </Button>
  );
}
