import { cn } from "@/lib/utils";
import { NightPlanType, TimeEntriesBySite } from "../../types";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useState } from "react";
import RealTimeEntry from "../Results/RealTimeEntry";

export default function Result({
  plans,
  label,
  site,
  onSiteChange,
  plotHeight,
  tablesOpen,
  onTablesOpenChange,
  className,
}: {
  plans: NightPlanType[];
  /** Shown before the event description, e.g. "Current plan". */
  label?: string;
  /** The selected site tab, when the parent controls it. */
  site?: string;
  onSiteChange?: (site: string) => void;
  /** Height of the plot in pixels; the plot's default when not given. */
  plotHeight?: number;
  /** See RealTimeEntry: makes the observation tables collapsible. */
  tablesOpen?: boolean;
  onTablesOpenChange?: (open: boolean) => void;
  className?: string;
}) {
  // Used when the parent does not control the site tab.
  const [ownSite, setOwnSite] = useState("GN");

  if (!plans || plans.length === 0) {
    return null;
  }

  const plan = plans.at(-1)!;
  const sites = plan.timeEntriesBySite.map((en) => en.site as string);
  // Fall back to the first site the plan has, so a GS-only plan still shows.
  const preferred = site ?? ownSite;
  const selectedSite = sites.includes(preferred) ? preferred : sites[0];

  return (
    <div
      className={cn(
        "border rounded-md flex flex-col gap-2 p-3 flex-wrap",
        "dark:bg-white/20 bg-black/10",
        className,
      )}
    >
      <h1 className="font-bold w-full">
        {label && <span>{label}: </span>}
        {plan.timeEntriesBySite[0].timeEntries.at(-1).event.description}
      </h1>
      <Tabs
        value={selectedSite}
        onValueChange={onSiteChange ?? setOwnSite}
        className="gap-0 w-full"
      >
        <TabsList
          className={cn("p-0 rounded-br-none rounded-bl-none", "bg-tranparent")}
        >
          {plan.timeEntriesBySite.map((en: TimeEntriesBySite, idx: number) => (
            <TabsTrigger
              key={`siteTrigger${idx}`}
              value={en.site}
              className={cn(
                "rounded-br-none rounded-bl-none border border-b-0",
                "dark:border-white/20 border-black/20",
                "dark:data-[state=active]:bg-black/40 data-[state=active]:bg-white/40",
                "data-[state=active]:border-b-0 data-[state=active]:outline-0",
              )}
            >
              {en.site}
            </TabsTrigger>
          ))}
        </TabsList>
        {plan.timeEntriesBySite.map((en: TimeEntriesBySite, idx: number) => (
          <TabsContent
            key={`siteContent${idx}`}
            value={en.site}
            className={cn(
              "bg-white/40 dark:bg-black/40",
              "p-4 border w-full",
              "data-[state=active]:border-t-0 data-[state=active]:outline-0",
              "border-black/20 dark:border-white/20",
              "rounded-tr-md rounded-tl-none rounded-br-md rounded-bl-md",
            )}
          >
            <RealTimeEntry
              timeEntry={en.timeEntries.at(-1)}
              eveTwilight={en.eveTwilight}
              mornTwilight={en.mornTwilight}
              nightStart={en.nightStart}
              nightEnd={en.nightEnd}
              moonAltitude={en.moonAltitude}
              moonIllumination={en.moonIllumination}
              moonWaxing={en.moonWaxing}
              site={en.site}
              plotHeight={plotHeight}
              tablesOpen={tablesOpen}
              onTablesOpenChange={onTablesOpenChange}
            />
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
}
