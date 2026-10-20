import { useContext, useEffect } from "react";
import { cn } from "@/lib/utils";
import { RunButton } from "./RunButton";
import { GlobalStateContext } from "../GlobalState/GlobalState";
import { useReactiveVar } from "@apollo/client";
import { isRealtimeConnectedVar } from "@/apollo-client";
import { useLazyQuery } from "@apollo/client";
import { scheduleV2Query, onDemandQuery } from "@/components/Operation/query";

export default function OnDemandControl() {
  const { loadingPlan, setLoadingPlan } = useContext(GlobalStateContext);

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
        "border rounded-md grid grid-cols-2 gap-2 p-3 w-full",
        "dark:bg-white/20 bg-black/10",
      )}
    >
      <RunButton
        loadingPlan={loadingPlan}
        run={runPlan}
        isRunDisabled={loadingPlan}
        title="RUN FROM START OF NIGHT"
        full={true}
      />

      <RunButton
        loadingPlan={loadingPlan}
        run={onDemandPlan}
        isRunDisabled={loadingPlan}
        title="RUN FROM NOW"
        full={true}
      />
    </div>
  );
}
