import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { RunButton } from "../ControlPanel/RunButton";
import { ProgramSelectorDialog } from "../ControlPanel/ProgramSelectorDialog";
import { VisibilityRange } from "../ControlPanel/VisibilityRange";
import { DateTimeSelector } from "../ControlPanel/DateTimeSelector";
import { DateRange } from "react-day-picker";
import { FaCog, FaTrash } from "react-icons/fa";
import {
  getSiteOffset,
  toDateOnlyString,
  toUtcIsoString,
  tzDateToString,
  utcToLocal,
} from "@/helpers/utcTime";
import { useMutation, useQuery } from "@apollo/client";
import { updateBuildParameters } from "./mutation";
import { SiteNightTimesEntry } from "@/gql/graphql";
import { getProgramList } from "./query";
import DisplayBuildParameters from "./DisplayBuildParameters";

export default function BuildParameters({
  vertical = false,
}: {
  vertical?: boolean;
}) {
  const [buildParams] = useMutation(updateBuildParameters, {
    fetchPolicy: "no-cache",
    context: { clientName: "realtimeClient" },
  });

  const [date, setDate] = useState<DateRange | undefined>({
    from: undefined,
    to: undefined,
  });
  const [programs, updatePrograms] = useState([]);
  const [startTimeGN, setStartTimeGN] = useState<Date | undefined>(undefined);
  const [startTimeGS, setStartTimeGS] = useState<Date | undefined>(undefined);
  const [endTimeGN, setEndTimeGN] = useState<Date | undefined>(undefined);
  const [endTimeGS, setEndTimeGS] = useState<Date | undefined>(undefined);
  // Optional. Left empty, the backend starts the simulated clock at the built
  // night's evening twilight; set it to jump straight into the night instead.
  const [simulatedNow, setSimulatedNow] = useState<Date | undefined>(undefined);

  // The list must match the night being built, not today. Same order the
  // backend uses: visibility start first, then the earliest night start.
  const nightDate =
    date?.from ??
    [startTimeGN, startTimeGS]
      .filter((d): d is Date => Boolean(d))
      .sort((a, b) => a.getTime() - b.getTime())[0];

  const { data: programList, loading: programListLoading } = useQuery(
    getProgramList,
    {
      variables: { nightDate: nightDate ? toDateOnlyString(nightDate) : null },
      fetchPolicy: "no-cache",
      context: { clientName: "realtimeClient" },
    }
  );

  useEffect(() => {
    if (programList) {
      updatePrograms((current) => {
        // A refetch (new night, reconnect) must not silently re-check what the
        // operator unchecked: keep their picks, default unseen programs to on.
        const picked = new Map(current.map((p) => [p.id, p.checked]));
        return programList.availablePrograms.map((p) => ({
          label: p.refLabel,
          id: p.id,
          checked: picked.get(p.id) ?? true,
          disabled: false,
        }));
      });
    }
  }, [programList]);

  function resetPrograms() {
    updatePrograms(
      programList.availablePrograms.map((p) => ({
        label: p.refLabel,
        id: p.id,
        checked: true,
        disabled: false,
      }))
    );
  }

  function setProgram(program: string, state: boolean) {
    const auxProgramList = [...programs];
    auxProgramList.find((p) => p.id === program).checked = state;
    updatePrograms(auxProgramList);
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
          simulatedNow: simulatedNow ? toUtcIsoString(simulatedNow) : undefined,
        },
      },
    });
  }

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
                className={cn(startTimeGN ? "text-red-500 cursor-pointer" : "")}
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
                className={cn(endTimeGN ? "text-red-500 cursor-pointer" : "")}
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
                className={cn(startTimeGS ? "text-red-500 cursor-pointer" : "")}
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
          <DateTimeSelector
            title="Simulated Now (UT)"
            dateTime={simulatedNow!}
            setDateTime={setSimulatedNow}
            setToNow={() => {}}
            setToNowButton={false}
            vertical={vertical}
            clearButton={
              <FaTrash
                className={cn(simulatedNow ? "text-red-500 cursor-pointer" : "")}
                onClick={() => setSimulatedNow(undefined)}
              />
            }
          />
          <ProgramSelectorDialog
            programs={programs}
            setProgram={setProgram}
            resetPrograms={resetPrograms}
            validationMode={false}
            full={true}
            loading={programListLoading}
          />
          <RunButton
            loadingPlan={false}
            run={sendBuildParams}
            // An empty programList reads as "no filter" and the build loads every
            // program of the current day, so never send one.
            isRunDisabled={
              programListLoading ||
              programs.filter((p) => p.checked).length === 0
            }
            title="Send Parameters"
            icon={<FaCog />}
            full={true}
          />
        </div>
        <DisplayBuildParameters />
      </div>
    </div>
  );
}
