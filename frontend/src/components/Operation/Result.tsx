import { cn } from "@/lib/utils";
import { NightPlanType, TimeEntriesBySite } from "../../types";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import RealTimeEntry from "../Results/RealTimeEntry";

export default function Result({ plans }: { plans: NightPlanType[] }) {
  if (!plans || plans.length === 0) {
    return null;
  }

  const plan = plans.at(-1)!;

  return (
    <div
      className={cn(
        "border rounded-md flex flex-col gap-2 p-3 flex-wrap",
        "dark:bg-white/20 bg-black/10",
      )}
    >
      <h1 className="font-bold w-full">
        {plan.timeEntriesBySite[0].timeEntries.at(-1).event.description}
      </h1>
      <Tabs defaultValue="GN" className="gap-0 w-full">
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
              site={en.site}
            />
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
}
