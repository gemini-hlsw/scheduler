import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import RtPlan from "./RtPlan";
import { NightPlansWithEvent } from "@/../../schema/web/graphql";

export function Result({ rtPlan }: { rtPlan: NightPlansWithEvent }) {
  if (
    !rtPlan ||
    !rtPlan.nightPlans ||
    !rtPlan.nightPlans.plansPerSite ||
    rtPlan.nightPlans.plansPerSite.length === 0
  ) {
    return null;
  }

  return (
    <div
      className={cn(
        "border rounded-md flex flex-col gap-2 p-3 flex-wrap",
        "dark:bg-white/20 bg-black/10",
      )}
    >
      <h1 className="font-bold w-full">Plan result - {rtPlan.event}</h1>
      <Tabs defaultValue="GN" className="gap-0 w-full">
        <TabsList
          className={cn("p-0 rounded-br-none rounded-bl-none", "bg-tranparent")}
        >
          {rtPlan.nightPlans.plansPerSite.map((plan) => (
            <TabsTrigger
              key={`siteTrigger${plan.site}`}
              value={plan.site}
              className={cn(
                "rounded-br-none rounded-bl-none border border-b-0",
                "dark:border-white/20 border-black/20",
                "dark:data-[state=active]:bg-black/40 data-[state=active]:bg-white/40",
                "data-[state=active]:border-b-0 data-[state=active]:outline-0",
              )}
            >
              {plan.site}
            </TabsTrigger>
          ))}
        </TabsList>
        {rtPlan.nightPlans.plansPerSite.map((plan) => (
          <TabsContent
            key={`siteContent${plan.site}`}
            value={plan.site}
            className={cn(
              "bg-white/40 dark:bg-black/40",
              "p-4 border w-full",
              "data-[state=active]:border-t-0 data-[state=active]:outline-0",
              "border-black/20 dark:border-white/20",
              "rounded-tr-md rounded-tl-none rounded-br-md rounded-bl-md",
            )}
          >
            <RtPlan plan={plan} />
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
}
