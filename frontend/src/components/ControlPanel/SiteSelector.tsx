import { Label } from "@/components/ui/label";
import { Field } from "@/components/ui/field";
import { Button } from "@/components/ui/button";
import { ButtonGroup } from "@/components/ui/button-group";
import { cn } from "@/lib/utils";

export function SiteSelector({
  site,
  setSite,
  loadingPlan,
  showAll = false,
  vertical = false,
}: {
  site?: string;
  setSite?: (site: string) => void;
  loadingPlan?: boolean;
  showAll?: boolean;
  vertical?: boolean;
}) {
  return (
    <Field orientation="horizontal" className="w-fit">
      <Label className={cn("text-nowrap", vertical ? "w-32" : "w-fit")}>
        Site
      </Label>
      <ButtonGroup>
        <Button
          variant="default"
          size="sm"
          className={cn(
            "dark:text-white text-black",
            "dark:hover:bg-blue-700 hover:bg-blue-500",
            "px-2",
            site === "GN"
              ? "dark:bg-blue-800 bg-blue-400"
              : "dark:bg-white/20 bg-black/20"
          )}
          disabled={loadingPlan}
          onClick={() => setSite("GN")}
        >
          GN
        </Button>
        <Button
          variant="default"
          size="sm"
          className={cn(
            "dark:text-white text-black",
            "dark:hover:bg-blue-700 hover:bg-blue-500",
            "px-2",
            site === "GS"
              ? "dark:bg-blue-800 bg-blue-400"
              : "dark:bg-white/20 bg-black/20"
          )}
          disabled={loadingPlan}
          onClick={() => setSite("GS")}
        >
          GS
        </Button>
        {showAll && (
          <Button
            variant="default"
            size="sm"
            className={cn(
              "dark:text-white text-black",
              "dark:hover:bg-blue-700 hover:bg-blue-500",
              "px-2",
              site === "ALL_SITES"
                ? "dark:bg-blue-800 bg-blue-400"
                : "dark:bg-white/20 bg-black/20"
            )}
            disabled={loadingPlan}
            onClick={() => setSite("ALL_SITES")}
          >
            ALL
          </Button>
        )}
      </ButtonGroup>
    </Field>
  );
}
