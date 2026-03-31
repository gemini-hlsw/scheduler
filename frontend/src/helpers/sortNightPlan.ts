import { TimeEntryType } from "../types";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function sortNightPlan(timeline: any) {
  const newTimelines = [];
  for (const n_idx in timeline) {
    const sTimelineEntries = [];
    for (const site in timeline[n_idx]) {
      const sEntries: TimeEntryType[] = [];
      const eveTwi = timeline[n_idx][site][0].event.time;
      const mornTwi =
        timeline[n_idx][site][timeline[n_idx][site].length - 1].event.time;
      for (const entry in timeline[n_idx][site]) {
        if (Object.keys(timeline[n_idx][site][entry].plan).length > 0) {
          const planTime = timeline[n_idx][site][entry].plan.start.split(" ");
          sEntries.push({
            startTimeSlots: parseInt(
              timeline[n_idx][site][entry].startTimeSlot
            ),
            event: timeline[n_idx][site][entry].event.description,
            plan: {
              startTime: `${planTime[0]}T${planTime[1]}:00`,
              visits: timeline[n_idx][site][entry].plan.visits.map(
                (v: {
                  startTime: string;
                  endTime: string;
                  obs_class: string;
                }) => {
                  const vStartTime = v.startTime.split(" ");
                  const vEndTime = v.endTime.split(" ");
                  return {
                    ...v,
                    endTime: `${vEndTime[0]}T${vEndTime[1]}:00`,
                    startTime: `${vStartTime[0]}T${vStartTime[1]}:00`,
                    obsClass: v.obs_class,
                  };
                }
              ),
              nightStats: {
                timeLoss: timeline[n_idx][site][entry].plan.nightStats.timeLoss,
                planScore: parseFloat(
                  timeline[n_idx][site][entry].plan.nightStats.planScore
                ),
                nToos: 0,
                completionFraction:
                  timeline[n_idx][site][entry].plan.nightStats
                    .completionFraction,
                programCompletion:
                  timeline[n_idx][site][entry].plan.nightStats
                    .programCompletion,
              },
            },
          });
        }
      }
      sTimelineEntries.push({
        site: site as string,
        eveTwilight: eveTwi as string,
        mornTwilight: mornTwi as string,
        timeEntries: sEntries,
      });
    }
    newTimelines.push({
      nightIndex: parseInt(n_idx),
      timeEntriesBySite: sTimelineEntries,
    });
  }
  return newTimelines;
}
