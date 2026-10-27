import { Outlet } from "react-router-dom";
import Layout from "./components/Layout/Layout";
import { useQuery, useReactiveVar, useSubscription } from "@apollo/client";
import { isRealtimeConnectedVar } from "./apollo-client";
import { operationNightQuery } from "./components/Operation/query";
import { subscriptionQueueSchedule } from "./components/GlobalState/scheduleSubscription";
import { useContext, useEffect } from "react";
import { GlobalStateContext } from "./components/GlobalState/GlobalState";
import { Toaster } from "@/components/ui/sonner";
import { toast } from "sonner";
import { type RunSummary } from "./types";

function App() {
  const {
    setNightPlans,
    setPlansSummary,
    setLoadingPlan,
    setOperationNight,
    uuid,
  } = useContext(GlobalStateContext);

  const {
    data: scheduleData,
    loading: subscriptionLoading,
    error,
  } = useSubscription(subscriptionQueueSchedule, {
    variables: { scheduleId: uuid },
  });

  const {
    data: rtData,
    loading: rtLoading,
    error: rtError,
  } = useSubscription(subscriptionQueueSchedule, {
    variables: { scheduleId: "operation" },
    context: { clientName: "realtimeClient" },
  });

  // The default (twilight) night the operation server plans for. Asked on load
  // and after every reconnect, and polled so it rolls over to the next night.
  const isRealtimeConnected = useReactiveVar(isRealtimeConnectedVar);
  const { data: operationNightData, refetch: refetchOperationNight } =
    useQuery(operationNightQuery, {
      fetchPolicy: "no-cache",
      context: { clientName: "realtimeClient" },
      pollInterval: 10 * 60 * 1000,
    });

  useEffect(() => {
    if (isRealtimeConnected) refetchOperationNight();
  }, [isRealtimeConnected]);

  useEffect(() => {
    if (operationNightData) {
      setOperationNight(operationNightData.operationNight);
    }
  }, [operationNightData]);

  useEffect(() => {
    try {
      if (!subscriptionLoading) {
        if (
          scheduleData &&
          scheduleData.queueSchedule.__typename === "NewNightPlans"
        ) {
          setNightPlans(scheduleData.queueSchedule.nightPlans.nightTimeline);
          setPlansSummary(scheduleData.queueSchedule.plansSummary);
        } else if (
          scheduleData &&
          scheduleData.queueSchedule.__typename === "NightPlansError"
        ) {
          toast.error(scheduleData.queueSchedule.error, {
            closeButton: true,
            duration: Infinity,
          });
          setNightPlans([]);
          setPlansSummary({} as RunSummary);
        } else if (
          scheduleData &&
          scheduleData.queueSchedule.__typename === "NewPlansRT"
        ) {
          // Deprecated
          console.log("Deprecated method");
        } else {
          toast.error(
            error?.message ??
              "Unknown type, probbably due a mismatch between server and UI versions",
            {
              closeButton: true,
              duration: Infinity,
            },
          );
          setNightPlans([]);
          setPlansSummary({} as RunSummary);
        }
        setLoadingPlan(false);
      }
    } catch (e) {
      console.log(e);
    }
  }, [scheduleData, subscriptionLoading]);

  useEffect(() => {
    try {
      if (!rtLoading) {
        if (rtData && rtData.queueSchedule.__typename === "NewNightPlans") {
          setNightPlans(rtData.queueSchedule.nightPlans.nightTimeline);
          setPlansSummary(rtData.queueSchedule.plansSummary);
        } else if (
          rtData &&
          rtData.queueSchedule.__typename === "NightPlansError"
        ) {
          toast.error(rtData.queueSchedule.error, {
            closeButton: true,
            duration: Infinity,
          });
          setNightPlans([]);
          setPlansSummary({} as RunSummary);
        } else if (rtData && rtData.queueSchedule.__typename === "NewPlansRT") {
          // Deprecated
          console.log("Deprecated method");
        } else if (
          rtData &&
          rtData.queueSchedule.__typename === "NightPlansWithEvent"
        ) {
          // Deprecated
          console.log("Deprecated method");
        } else {
          toast.error(
            rtError?.message ??
              "Unknown type, probbably due a mismatch between server and UI versions",
            {
              closeButton: true,
              duration: Infinity,
            },
          );
          setNightPlans([]);
          setPlansSummary({} as RunSummary);
        }
        setLoadingPlan(false);
      }
    } catch (e) {
      console.log(e);
    }
  }, [rtData, rtLoading]);

  return (
    <Layout>
      <Outlet />
      <Toaster />
    </Layout>
  );
}

export default App;
