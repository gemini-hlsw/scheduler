import { useContext, useEffect, useRef, useState } from "react";
import { useMutation } from "@apollo/client";
import { FaCheck, FaTimes } from "react-icons/fa";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { PlanReview } from "../GlobalState/planReview";
import { RealtimeBackendContext } from "../GlobalState/RealtimeBackend";
import { decidePlanMutation } from "./mutation";
import Result from "./Result";

// While a new plan waits for a decision both plots are drawn this tall
// (instead of the plot's default), so the two fit on screen together.
const COMPARE_PLOT_HEIGHT = 260;

/**
 * The plan in effect and, when the backend has proposed a newer one, that plan
 * below it with Accept and Discard. Both show the same site tab so they can
 * be compared side by side in time.
 *
 * The decision goes to the backend, which applies it for every browser; the
 * panel changes when the backend announces it (see planReview.ts).
 */
export default function ReviewedResult({ review }: { review: PlanReview }) {
  const [site, setSite] = useState("GN");
  const { current, pending } = review;
  const { clientName, tab } = useContext(RealtimeBackendContext);
  const [decidePlan, { loading: deciding }] = useMutation(decidePlanMutation, {
    fetchPolicy: "no-cache",
    context: { clientName },
  });

  const decide = (accept: boolean) => {
    if (!pending) return;
    decidePlan({
      variables: { proposalId: pending.proposalId, accept },
    }).catch((e) => {
      toast.error(e.message ?? String(e), {
        closeButton: true,
        duration: Infinity,
        toasterId: tab,
      });
      // The backend refused it: someone decided first, or a newer plan
      // replaced it. A network error leaves it, so it can be tried again.
      if (e.graphQLErrors?.length) review.dropPending();
    });
  };
  const plotHeight = pending ? COMPARE_PLOT_HEIGHT : undefined;
  // While comparing, each plan's observation tables can be collapsed; both
  // start collapsed so the two plots sit close together.
  const [currentTablesOpen, setCurrentTablesOpen] = useState(false);
  const [newTablesOpen, setNewTablesOpen] = useState(false);

  // Bring each new plan into view, or it can arrive unnoticed below the fold.
  const newPlanRef = useRef<HTMLElement>(null);
  useEffect(() => {
    if (pending) {
      setCurrentTablesOpen(false);
      setNewTablesOpen(false);
      newPlanRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  }, [pending]);

  return (
    <div className="flex flex-col gap-2">
      <Result
        plans={current}
        label={pending ? "Current plan" : undefined}
        site={site}
        onSiteChange={setSite}
        plotHeight={plotHeight}
        tablesOpen={pending ? currentTablesOpen : undefined}
        onTablesOpenChange={setCurrentTablesOpen}
      />
      {pending && (
        <section
          ref={newPlanRef}
          aria-label="New plan"
          className={cn(
            "flex flex-col gap-2 p-2 rounded-md",
            "border-2 border-amber-500",
          )}
        >
          <div className="flex flex-row flex-wrap items-center gap-2 px-1">
            <span className="font-bold text-amber-600 dark:text-amber-400">
              New plan received
            </span>
            <span className="text-sm opacity-70">
              Compare it with the current plan, then accept it to replace the
              current plan or discard it. The decision applies to everyone.
            </span>
            <div className="ml-auto flex flex-row gap-2">
              <Button
                size="sm"
                disabled={deciding}
                onClick={() => decide(true)}
              >
                <FaCheck />
                Accept
              </Button>
              <Button
                size="sm"
                variant="outline"
                disabled={deciding}
                onClick={() => decide(false)}
              >
                <FaTimes />
                Discard
              </Button>
            </div>
          </div>
          {pending.plans.length === 0 ? (
            <p className="text-sm px-1">The new plan is empty.</p>
          ) : (
            <Result
              plans={pending.plans}
              label="New plan"
              site={site}
              onSiteChange={setSite}
              plotHeight={plotHeight}
              tablesOpen={newTablesOpen}
              onTablesOpenChange={setNewTablesOpen}
            />
          )}
        </section>
      )}
    </div>
  );
}
