import { useContext, useEffect, useMemo, useState } from "react";
import { cn } from "@/lib/utils";
import { RunButton } from "../ControlPanel/RunButton";
import { ProgramSelectorDialog } from "../ControlPanel/ProgramSelectorDialog";
import { VisibilityRange } from "../ControlPanel/VisibilityRange";
import { VisibilityNights } from "../ControlPanel/VisibilityNights";
import { DateTimeSelector } from "../ControlPanel/DateTimeSelector";
import { NightTimeInput, NightWindow } from "../ControlPanel/NightTimeInput";
import { GlobalStateContext } from "../GlobalState/GlobalState";
import { toast } from "sonner";
import { DateRange } from "react-day-picker";
import { FaCog, FaTrash } from "react-icons/fa";
import {
  getSiteOffset,
  SITE_TIME_ZONES,
  toDateOnlyString,
  toSiteWallClock,
  toUtcIsoString,
  tzDateToString,
  utcToLocal,
} from "@/helpers/utcTime";
import { useMutation, useQuery, useSubscription } from "@apollo/client";
import {
  resetProgramSelection,
  setProgramSelection,
  updateBuildParameters,
} from "./mutation";
import { programSelectionSubscription } from "./subscription";
import { SiteNightTimesEntry } from "@/gql/graphql";
import { getProgramList } from "./query";
import DisplayBuildParameters, {
  SHOW_SIMULATED_NOW,
} from "./DisplayBuildParameters";

export default function BuildParameters({
  vertical = false,
  // Operations always plan tonight, so only a simulation picks the visibility
  // range; operations pick how many nights from tonight it covers instead.
  showVisibility = true,
  // Program Selection and Send Parameters side by side under the whole panel,
  // instead of at the bottom of the inputs column.
  buttonsRow = false,
  // Night start and end as times of day inside the operation server's
  // default night (see operationNight), instead of free dates and times.
  operationNightTimes = false,
}: {
  vertical?: boolean;
  showVisibility?: boolean;
  buttonsRow?: boolean;
  operationNightTimes?: boolean;
}) {
  const [buildParams] = useMutation(updateBuildParameters, {
    fetchPolicy: "no-cache",
    context: { clientName: "realtimeClient" },
  });

  const [date, setDate] = useState<DateRange | undefined>({
    from: undefined,
    to: undefined,
  });
  const [startTimeGN, setStartTimeGN] = useState<Date | undefined>(undefined);
  const [startTimeGS, setStartTimeGS] = useState<Date | undefined>(undefined);
  const [endTimeGN, setEndTimeGN] = useState<Date | undefined>(undefined);
  const [endTimeGS, setEndTimeGS] = useState<Date | undefined>(undefined);
  // Optional. Left empty, the backend starts the simulated clock at the built
  // night's evening twilight; set it to jump straight into the night instead.
  const [simulatedNow, setSimulatedNow] = useState<Date | undefined>(undefined);
  // Undefined runs the visibility to the end of the semester.
  const [visibilityNights, setVisibilityNights] = useState<number | undefined>(
    undefined
  );

  // The times the backend accepts for a custom night: strictly between the
  // twilights, which it rounds up to the minute.
  const { operationNight } = useContext(GlobalStateContext);
  const nightWindows = useMemo(() => {
    const minute = 60 * 1000;
    const ceilMinute = (iso: string) =>
      Math.ceil(new Date(iso).getTime() / minute) * minute;
    const windows: Record<string, NightWindow> = {};
    for (const { site, start, end } of operationNight) {
      const timeZone = SITE_TIME_ZONES[site];
      windows[site] = {
        from: toSiteWallClock(new Date(ceilMinute(start) + minute), timeZone),
        to: toSiteWallClock(new Date(ceilMinute(end) - minute), timeZone),
      };
    }
    return windows;
  }, [operationNight]);

  // Once the night rolls over, times picked for the previous one no longer fit.
  useEffect(() => {
    if (!operationNightTimes) return;
    const fits = (site: string, d: Date | undefined) =>
      !d ||
      (nightWindows[site] !== undefined &&
        nightWindows[site].from <= d &&
        d <= nightWindows[site].to);
    if (!fits("GN", startTimeGN)) setStartTimeGN(undefined);
    if (!fits("GN", endTimeGN)) setEndTimeGN(undefined);
    if (!fits("GS", startTimeGS)) setStartTimeGS(undefined);
    if (!fits("GS", endTimeGS)) setEndTimeGS(undefined);
  }, [nightWindows]);

  function setNightStartToNow(site: string, setTime: (d: Date) => void) {
    const now = toSiteWallClock(new Date(), SITE_TIME_ZONES[site]);
    now.setSeconds(0, 0);
    const window = nightWindows[site];
    if (window && window.from <= now && now <= window.to) {
      setTime(now);
    } else {
      toast.error(`${site} night has not started yet, or is already over.`);
    }
  }

  // The list must match the night being built, not today. Same order the
  // backend uses: visibility start first, then the earliest night start.
  const nightDate =
    date?.from ??
    [startTimeGN, startTimeGS]
      .filter((d): d is Date => Boolean(d))
      .sort((a, b) => a.getTime() - b.getTime())[0];

  const {
    data: programList,
    loading: programListLoading,
    refetch: refetchProgramList,
  } = useQuery(getProgramList, {
    variables: { nightDate: nightDate ? toDateOnlyString(nightDate) : null },
    fetchPolicy: "no-cache",
    context: { clientName: "realtimeClient" },
  });

  // The selection lives on the server and is shared by every browser: each
  // change goes up as a mutation and comes back to all of them through this
  // subscription, which also sends the current selection when it connects.
  const { data: selectionData, loading: selectionLoading } = useSubscription(
    programSelectionSubscription,
    { context: { clientName: "realtimeClient" } }
  );
  const [setSelection] = useMutation(setProgramSelection, {
    fetchPolicy: "no-cache",
    context: { clientName: "realtimeClient" },
  });
  const [resetSelection] = useMutation(resetProgramSelection, {
    fetchPolicy: "no-cache",
    context: { clientName: "realtimeClient" },
  });

  const programs = useMemo(() => {
    // The server keeps the unchecked programs, so ones it has not seen yet
    // (new night, new in the ODB) start checked.
    const deselected = new Set(
      selectionData?.programSelectionUpdates.deselected ?? []
    );
    return (programList?.availablePrograms ?? []).map((p) => ({
      label: p.refLabel,
      id: p.id,
      checked: !deselected.has(p.id),
      disabled: false,
    }));
  }, [programList, selectionData]);

  function resetPrograms() {
    resetSelection();
  }

  function setPrograms(programIds: string[], state: boolean) {
    setSelection({ variables: { programIds, selected: state } });
  }

  function setProgram(program: string, state: boolean) {
    setPrograms([program], state);
  }

  function sendBuildParams() {
    const nightTimes = [];
    if (endTimeGN || startTimeGN) {
      nightTimes.push({
        site: "GN",
        nightTimes: {
          nightEnd: endTimeGN
            ? tzDateToString(endTimeGN, "Pacific/Honolulu")
            : undefined,
          nightStart: startTimeGN
            ? tzDateToString(startTimeGN, "Pacific/Honolulu")
            : undefined,
        },
      } as SiteNightTimesEntry);
    }

    if (endTimeGS || startTimeGS) {
      nightTimes.push({
        site: "GS",
        nightTimes: {
          nightEnd: endTimeGS
            ? tzDateToString(endTimeGS, "America/Santiago")
            : undefined,
          nightStart: startTimeGS
            ? tzDateToString(startTimeGS, "America/Santiago")
            : undefined,
        },
      } as SiteNightTimesEntry);
    }

    buildParams({
      variables: {
        buildParamsInput: {
          nightTimes: nightTimes.length ? nightTimes : undefined,
          programList: programs.filter((p) => p.checked).map((p) => p.id),
          visibilityEnd: date?.to ? toUtcIsoString(date.to) : undefined,
          visibilityStart: date?.from ? toUtcIsoString(date.from) : undefined,
          visibilityNights: showVisibility ? undefined : visibilityNights,
          simulatedNow: simulatedNow ? toUtcIsoString(simulatedNow) : undefined,
        },
      },
    });
  }

  const buttons = (
    <>
      <ProgramSelectorDialog
        programs={programs}
        setProgram={setProgram}
        setPrograms={setPrograms}
        resetPrograms={resetPrograms}
        validationMode={false}
        full={true}
        loading={programListLoading || selectionLoading}
        // Programs can leave the ODB while the night stays the same, and the
        // query only reruns when nightDate changes: read the latest on open.
        onOpen={() => refetchProgramList()}
      />
      <RunButton
        loadingPlan={false}
        run={sendBuildParams}
        // An empty programList reads as "no filter" and the build loads every
        // program of the current day, so never send one.
        // Until the shared selection arrives every program reads as checked.
        isRunDisabled={
          programListLoading ||
          selectionLoading ||
          programs.filter((p) => p.checked).length === 0
        }
        title="Send Parameters"
        icon={<FaCog />}
        full={true}
      />
    </>
  );

  return (
    <div
      className={cn(
        "border rounded-md flex flex-col gap-2 p-3 flex-wrap",
        "dark:bg-white/20 bg-black/10"
      )}
    >
      <div className="flex flex-row gap-2">
        <div
          className={cn(
            "flex gap-1 items-center",
            vertical ? "flex-col" : "flex-row"
          )}
        >
          <h1 className="font-bold self-start">Build Parameters</h1>
          {showVisibility ? (
            <VisibilityRange
              date={date}
              setDate={setDate}
              vertical={vertical}
              clearButton={
                <FaTrash
                  className={cn(
                    date?.to || date?.from ? "text-red-500 cursor-pointer" : ""
                  )}
                  onClick={() => setDate({ from: undefined, to: undefined })}
                />
              }
            />
          ) : (
            <VisibilityNights
              nights={visibilityNights}
              setNights={setVisibilityNights}
              vertical={vertical}
            />
          )}
          {operationNightTimes ? (
            <>
              <NightTimeInput
                title="GN HST Night Start"
                dateTime={startTimeGN}
                setDateTime={setStartTimeGN}
                window={nightWindows.GN}
                setToNow={() => setNightStartToNow("GN", setStartTimeGN)}
                vertical={vertical}
                clearButton={
                  <FaTrash
                    className={cn(
                      startTimeGN ? "text-red-500 cursor-pointer" : ""
                    )}
                    onClick={() => setStartTimeGN(undefined)}
                  />
                }
              />
              <NightTimeInput
                title="GN HST Night End"
                dateTime={endTimeGN}
                setDateTime={setEndTimeGN}
                window={nightWindows.GN}
                vertical={vertical}
                clearButton={
                  <FaTrash
                    className={cn(
                      endTimeGN ? "text-red-500 cursor-pointer" : ""
                    )}
                    onClick={() => setEndTimeGN(undefined)}
                  />
                }
              />
              <NightTimeInput
                title="GS CLT Night Start"
                dateTime={startTimeGS}
                setDateTime={setStartTimeGS}
                window={nightWindows.GS}
                setToNow={() => setNightStartToNow("GS", setStartTimeGS)}
                vertical={vertical}
                clearButton={
                  <FaTrash
                    className={cn(
                      startTimeGS ? "text-red-500 cursor-pointer" : ""
                    )}
                    onClick={() => setStartTimeGS(undefined)}
                  />
                }
              />
              <NightTimeInput
                title="GS CLT Night End"
                dateTime={endTimeGS}
                setDateTime={setEndTimeGS}
                window={nightWindows.GS}
                vertical={vertical}
                clearButton={
                  <FaTrash
                    className={cn(
                      endTimeGS ? "text-red-500 cursor-pointer" : ""
                    )}
                    onClick={() => setEndTimeGS(undefined)}
                  />
                }
              />
            </>
          ) : (
            <>
              <DateTimeSelector
                title="GN HST Night Start"
                dateTime={startTimeGN!}
                setDateTime={setStartTimeGN}
                setToNow={() => {
                  setStartTimeGN(
                    new Date(
                      new Date(utcToLocal(new Date(), getSiteOffset("GN")))
                        .toISOString()
                        .split(".")[0]
                        .replace("T", " ")
                    )
                  );
                }}
                setToNowButton={true}
                vertical={vertical}
                clearButton={
                  <FaTrash
                    className={cn(
                      startTimeGN ? "text-red-500 cursor-pointer" : ""
                    )}
                    onClick={() => setStartTimeGN(undefined)}
                  />
                }
              />
              <DateTimeSelector
                title="GN HST Night End"
                dateTime={endTimeGN!}
                setDateTime={setEndTimeGN}
                setToNow={() => {}}
                setToNowButton={false}
                vertical={vertical}
                clearButton={
                  <FaTrash
                    className={cn(
                      endTimeGN ? "text-red-500 cursor-pointer" : ""
                    )}
                    onClick={() => setEndTimeGN(undefined)}
                  />
                }
              />
              <DateTimeSelector
                title="GS CLT Night Start"
                dateTime={startTimeGS!}
                setDateTime={setStartTimeGS}
                setToNow={() => {
                  setStartTimeGS(
                    new Date(
                      new Date(utcToLocal(new Date(), getSiteOffset("GS")))
                        .toISOString()
                        .split(".")[0]
                        .replace("T", " ")
                    )
                  );
                }}
                setToNowButton={true}
                vertical={vertical}
                clearButton={
                  <FaTrash
                    className={cn(
                      startTimeGS ? "text-red-500 cursor-pointer" : ""
                    )}
                    onClick={() => setStartTimeGS(undefined)}
                  />
                }
              />
              <DateTimeSelector
                title="GS CLT Night End"
                dateTime={endTimeGS!}
                setDateTime={setEndTimeGS}
                setToNow={() => {}}
                setToNowButton={false}
                vertical={vertical}
                clearButton={
                  <FaTrash
                    className={cn(
                      endTimeGS ? "text-red-500 cursor-pointer" : "left-auto"
                    )}
                    onClick={() => setEndTimeGS(undefined)}
                  />
                }
              />
            </>
          )}
          {SHOW_SIMULATED_NOW && (
            <DateTimeSelector
              title="Simulated Now (UT)"
              dateTime={simulatedNow!}
              setDateTime={setSimulatedNow}
              setToNow={() => {}}
              setToNowButton={false}
              vertical={vertical}
              clearButton={
                <FaTrash
                  className={cn(
                    simulatedNow ? "text-red-500 cursor-pointer" : ""
                  )}
                  onClick={() => setSimulatedNow(undefined)}
                />
              }
            />
          )}
          {!buttonsRow && buttons}
        </div>
        <DisplayBuildParameters showVisibility={showVisibility} />
      </div>
      {buttonsRow && <div className="grid grid-cols-2 gap-2">{buttons}</div>}
    </div>
  );
}
