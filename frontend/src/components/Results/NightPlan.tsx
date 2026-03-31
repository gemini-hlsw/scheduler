import { NightPlanType, TimeEntriesBySite } from "../../types";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import EntryBySite from "./EntryBySite";
import { cn } from "@/lib/utils";

export default function NightPlan({ nightPlan }: { nightPlan: NightPlanType }) {
  return (
    <Tabs defaultValue="GN" className="gap-0">
      <TabsList
        className={cn("p-0 rounded-br-none rounded-bl-none", "bg-tranparent")}
      >
        {nightPlan.timeEntriesBySite.map(
          (en: TimeEntriesBySite, idx: number) => (
            <TabsTrigger
              key={`siteTrigger${idx}`}
              value={en.site}
              className={cn(
                "rounded-br-none rounded-bl-none border border-b-0",
                "dark:border-white/20 border-black/20",
                "dark:data-[state=active]:bg-black/40 data-[state=active]:bg-white/40",
                "data-[state=active]:border-b-0 data-[state=active]:outline-0"
              )}
            >
              {en.site}
            </TabsTrigger>
          )
        )}
      </TabsList>
      {nightPlan.timeEntriesBySite.map((en: TimeEntriesBySite, idx: number) => (
        <TabsContent
          key={`siteContent${idx}`}
          value={en.site}
          className={cn(
            "bg-white/40 dark:bg-black/40",
            "p-4 border w-full",
            "data-[state=active]:border-t-0 data-[state=active]:outline-0",
            "border-black/20 dark:border-white/20",
            "rounded-tr-md rounded-tl-none rounded-br-md rounded-bl-md"
          )}
        >
          <EntryBySite entryBySite={en} key={`entrySite${idx}`} />
        </TabsContent>
      ))}
    </Tabs>
  );
}
