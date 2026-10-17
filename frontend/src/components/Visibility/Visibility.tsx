import { useState } from "react";
import { cn } from "@/lib/utils";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import AggregatorStatusCard from "./AggregatorStatusCard";
import CoverageSummary from "./CoverageSummary";
import CoverageList from "./CoverageList";
import VisibleTonight from "./VisibleTonight";

/**
 * Visibility status for the realtime deployment: whether the Sight DB holds
 * everything the ODB expects, what the aggregator is doing, and what is
 * observable tonight.
 */
export default function Visibility() {
  // Bumped by the summary's Refresh so the list re-reads too; otherwise it would
  // keep showing the previous read until its own filters change.
  const [coverageRefresh, setCoverageRefresh] = useState(0);
  return (
    <div className="flex flex-col gap-3">
      <AggregatorStatusCard />
      <Tabs defaultValue="coverage" className="w-full">
        <TabsList>
          <TabsTrigger value="coverage">Coverage</TabsTrigger>
          <TabsTrigger value="tonight">Tonight</TabsTrigger>
        </TabsList>
        <TabsContent value="coverage" className={cn("flex flex-col gap-3")}>
          <CoverageSummary onRefresh={() => setCoverageRefresh((n) => n + 1)} />
          <CoverageList refreshKey={coverageRefresh} />
        </TabsContent>
        <TabsContent value="tonight">
          <VisibleTonight />
        </TabsContent>
      </Tabs>
    </div>
  );
}
