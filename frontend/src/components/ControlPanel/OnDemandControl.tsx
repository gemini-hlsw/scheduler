import { useContext, useEffect } from "react";
import { cn } from "@/lib/utils";
import { RunButton } from "./RunButton";
import { GlobalStateContext } from "../GlobalState/GlobalState";
import { useReactiveVar } from "@apollo/client";
import { isRealtimeConnectedVar } from "@/apollo-client";
import { useLazyQuery } from "@apollo/client";
import { scheduleV2Query, onDemandQuery } from "@/components/Operation/query";
import PlanStatus, {
  isPlanBusy,
  usePlanStatus,
} from "@/components/Operation/PlanStatus";

export default function OnDemandControl({
  // Operations only ever plan from the current time; replaying the night from
  // its start is a simulation tool.
  showStartOfNight = true,
}: {
  showStartOfNight?: boolean;
}) {
  const { loadingPlan, setLoadingPlan } = useContext(GlobalStateContext);
  // The server may be planning for another browser or for an event, which the
  // local flag knows nothing about.
  const planStatus = usePlanStatus();
  const planning = loadingPlan || isPlanBusy(planStatus);

  const [scheduleV2] = useLazyQuery(scheduleV2Query, {
    fetchPolicy: "no-cache",
    context: { clientName: "realtimeClient" },
  });

  const [onDemand] = useLazyQuery(onDemandQuery, {
    fetchPolicy: "no-cache",
    context: { clientName: "realtimeClient" },
  });

  function runPlan() {
    setLoadingPlan(true);
    scheduleV2();
  }

  function onDemandPlan() {
    setLoadingPlan(true);
    onDemand();
  }

  const { setConnectionState } = useContext(GlobalStateContext);
  const isRealtimeConnected = useReactiveVar(isRealtimeConnectedVar);

  useEffect(() => {
    setConnectionState({
      name: `Operation`,
      isConnected: isRealtimeConnected,
    });
  }, [isRealtimeConnected]);

  return (
    <div
      className={cn(
        "border rounded-md grid gap-2 p-3 w-full",
        showStartOfNight ? "grid-cols-2" : "grid-cols-1",
        "dark:bg-white/20 bg-black/10",
      )}
    >
      {showStartOfNight && (
        <RunButton
          loadingPlan={planning}
          run={runPlan}
          isRunDisabled={planning}
          title="RUN FROM START OF NIGHT"
          full={true}
        />
      )}

      <RunButton
        loadingPlan={planning}
        run={onDemandPlan}
        isRunDisabled={planning}
        title="GENERATE PLAN"
        full={true}
      />
      <div className={showStartOfNight ? "col-span-2" : ""}>
        <PlanStatus status={planStatus} />
      </div>
    </div>
  );
}
