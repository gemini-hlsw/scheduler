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
import { type PlanReview } from "./components/GlobalState/planReview";
import { type QueueScheduleSubscription } from "./gql/graphql";

/**
 * Apply a message from a live backend's plan subscription to its plan review.
 * Returns false for a message it does not know, so the caller can report it.
 */
function applyLivePlanMessage(
  message: QueueScheduleSubscription["queueSchedule"],
  review: PlanReview,
  toasterId: string,
): boolean {
  switch (message.__typename) {
    case "NewNightPlans":
      review.inEffect(message.nightPlans.nightTimeline);
      return true;
    case "ProposedNightPlans":
      review.proposed(message.proposalId, message.nightPlans.nightTimeline);
      return true;
    case "PlanDecision":
      review.decided(message.proposalId, message.accepted);
      toast(
        `New plan ${message.accepted ? "accepted" : "discarded"} by ${message.decidedBy}`,
        { closeButton: true, toasterId },
      );
      return true;
    case "NightPlansError":
      toast.error(message.error, {
        closeButton: true,
        duration: Infinity,
        toasterId,
      });
      review.dropPending();
      return true;
    case "NewPlansRT":
    case "NightPlansWithEvent":
      // Deprecated
      console.log("Deprecated method");
      return true;
    default:
      return false;
  }
}

function App() {
  const {
    setNightPlans,
    rtPlanReview,
    simPlanReview,
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
  const { data: operationNightData, refetch: refetchOperationNight } = useQuery(
    operationNightQuery,
    {
      fetchPolicy: "no-cache",
      context: { clientName: "realtimeClient" },
      pollInterval: 10 * 60 * 1000,
    },
  );

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
        if (
          !rtData ||
          !applyLivePlanMessage(
            rtData.queueSchedule,
            rtPlanReview,
            OPERATION_BACKEND.tab,
          )
        ) {
          toast.error(
            rtError?.message ??
              "Unknown type, probably due a mismatch between server and UI versions",
            {
              closeButton: true,
              duration: Infinity,
              toasterId: OPERATION_BACKEND.tab,
            },
          );
          rtPlanReview.dropPending();
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
        if (
          !simData ||
          !applyLivePlanMessage(
            simData.queueSchedule,
            simPlanReview,
            SIMULATION_BACKEND.tab,
          )
        ) {
          toast.error(
            simError?.message ??
              "Unknown type, probably due a mismatch between server and UI versions",
            {
              closeButton: true,
              duration: Infinity,
              toasterId: SIMULATION_BACKEND.tab,
            },
          );
          simPlanReview.dropPending();
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
