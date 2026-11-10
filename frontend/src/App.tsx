import { Outlet, useLocation } from "react-router-dom";
import Layout from "./components/Layout/Layout";
import { useQuery, useReactiveVar, useSubscription } from "@apollo/client";
import { isRealtimeConnectedVar } from "./apollo-client";
import { operationNightQuery } from "./components/Operation/query";
import { subscriptionQueueSchedule } from "./components/GlobalState/scheduleSubscription";
import { useContext, useEffect } from "react";
import { GlobalStateContext } from "./components/GlobalState/GlobalState";
import {
  OPERATION_BACKEND,
  SIMULATION_BACKEND,
} from "./components/GlobalState/RealtimeBackend";
import { Toaster } from "@/components/ui/sonner";
import { toast } from "sonner";
import { type RunSummary } from "./types";

function App() {
  const {
    setNightPlans,
    setRtNightPlans,
    setSimNightPlans,
    setPlansSummary,
    setLoadingPlan,
    setRtLoadingPlan,
    setSimLoadingPlan,
    setOperationNight,
    uuid,
  } = useContext(GlobalStateContext);

  const { pathname } = useLocation();

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

  const {
    data: simData,
    loading: simLoading,
    error: simError,
  } = useSubscription(subscriptionQueueSchedule, {
    variables: { scheduleId: "operation" },
    // Same query and variables as the realtime one above: without this Apollo
    // merges the two and the sim plans come from the operation dyno.
    context: { clientName: "simulationClient", queryDeduplication: false },
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
            toasterId: "validation",
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
              toasterId: "validation",
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
          setRtNightPlans(rtData.queueSchedule.nightPlans.nightTimeline);
        } else if (
          rtData &&
          rtData.queueSchedule.__typename === "NightPlansError"
        ) {
          toast.error(rtData.queueSchedule.error, {
            closeButton: true,
            duration: Infinity,
            toasterId: OPERATION_BACKEND.tab,
          });
          setRtNightPlans([]);
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
              "Unknown type, probably due a mismatch between server and UI versions",
            {
              closeButton: true,
              duration: Infinity,
              toasterId: OPERATION_BACKEND.tab,
            },
          );
          setRtNightPlans([]);
        }
        setRtLoadingPlan(false);
      }
    } catch (e) {
      console.log(e);
    }
  }, [rtData, rtLoading]);

  useEffect(() => {
    try {
      if (!simLoading) {
        if (simData && simData.queueSchedule.__typename === "NewNightPlans") {
          setSimNightPlans(simData.queueSchedule.nightPlans.nightTimeline);
        } else if (
          simData &&
          simData.queueSchedule.__typename === "NightPlansError"
        ) {
          toast.error(simData.queueSchedule.error, {
            closeButton: true,
            duration: Infinity,
            toasterId: SIMULATION_BACKEND.tab,
          });
          setSimNightPlans([]);
        } else if (
          simData &&
          (simData.queueSchedule.__typename === "NewPlansRT" ||
            simData.queueSchedule.__typename === "NightPlansWithEvent")
        ) {
          // Deprecated
          console.log("Deprecated method");
        } else {
          toast.error(
            simError?.message ??
              "Unknown type, probably due a mismatch between server and UI versions",
            {
              closeButton: true,
              duration: Infinity,
              toasterId: SIMULATION_BACKEND.tab,
            },
          );
          setSimNightPlans([]);
        }
        setSimLoadingPlan(false);
      }
    } catch (e) {
      console.log(e);
    }
  }, [simData, simLoading]);

  // One toaster per tab, and every toast names its tab with toasterId.
  // Gotcha: there is no default toaster on purpose, so a toast without a
  // toasterId is never shown. Toasters off the current tab are hidden, not
  // unmounted: sonner does not replay toasts to a toaster mounted later, so
  // an error raised while on another tab would be lost.
  const toasterTabs = [
    { id: "validation", path: "/validation" },
    { id: OPERATION_BACKEND.tab, path: "/operation" },
    { id: SIMULATION_BACKEND.tab, path: "/simulation" },
  ];

  return (
    <Layout>
      <Outlet />
      {toasterTabs.map(({ id, path }) => (
        <div key={id} className={pathname === path ? undefined : "invisible"}>
          <Toaster id={id} />
        </div>
      ))}
    </Layout>
  );
}

export default App;
