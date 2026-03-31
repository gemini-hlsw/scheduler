import { useContext, useEffect, useState } from "react";
import { ProgramSelectorDialog } from "./ProgramSelectorDialog";
import { ProgramListType } from "./ProgramSelection/ProgramList";
import { toUtcIsoString } from "../../helpers/utcTime";
import { cn } from "@/lib/utils";
import { addDays } from "date-fns/addDays";
import { DateRange } from "react-day-picker";
import { SiteSelector } from "./SiteSelector";
import { VisibilityRange } from "./VisibilityRange";
import { DateTimeSelector } from "./DateTimeSelector";
import { RunButton } from "./RunButton";
import { getDefaultDate } from "@/helpers/defaultDate";
import { SemesterVisibility } from "./SemesterVisibility";
import { NightsNumber } from "./NightsNumber";
import { GlobalStateContext } from "../GlobalState/GlobalState";
import { useReactiveVar } from "@apollo/client";
import { isSchedulerConnectedVar } from "@/apollo-client";

export default function ControlPanel({
  loadingPlan,
  runPlan,
  programList,
  vertical = false,
  validationMode = false,
}: {
  loadingPlan: boolean;
  // eslint-disable-next-line @typescript-eslint/no-unsafe-function-type
  runPlan: Function;
  programList: ProgramListType[];
  vertical?: boolean;
  validationMode?: boolean;
}) {
  const { setConnectionState, uuid } = useContext(GlobalStateContext);
  const isOperationConnected = useReactiveVar(isSchedulerConnectedVar);

  useEffect(() => {
    setConnectionState({
      name: uuid,
      isConnected: isOperationConnected,
    });
  }, [isOperationConnected]);

  useEffect(() => {
    updatePrograms(structuredClone(programList));
  }, [programList]);

  const DEFAULT_NIGHT_LENGTH_HOURS = 10;
  const defaultDate = getDefaultDate(validationMode);
  const [date, setDate] = useState<DateRange | undefined>({
    from: defaultDate,
    to: addDays(defaultDate, 10),
  });
  const [site, setSite] = useState(undefined);
  const [programs, updatePrograms] = useState(structuredClone(programList));
  const [numNight, setNumNight] = useState<number>(2);
  const [semesterVisibility, setSemesterVisibility] = useState<boolean>(false);
  const [startTime, setStartTime] = useState<Date | undefined>(defaultDate);
  const defaultEnd = new Date(defaultDate);
  defaultEnd.setHours(defaultEnd.getHours() + DEFAULT_NIGHT_LENGTH_HOURS);
  const [endTime, setEndTime] = useState<Date | null>(defaultEnd);

  useEffect(() => {
    updatePrograms(structuredClone(programList));
  }, [programList]);

  function setToNow() {
    const now = new Date();
    now.setHours(now.getHours() + now.getTimezoneOffset() / 60);
    setStartTime(new Date(toUtcIsoString(now)));
  }

  function setProgram(program: string, state: boolean) {
    const auxProgramList = [...programs];
    auxProgramList.find((p) => p.id === program).checked = state;
    updatePrograms(auxProgramList);
  }

  const isRunDisabled = !(
    site &&
    date !== undefined &&
    date.from !== undefined &&
    date.to !== undefined
  );

  return (
    <div
      className={cn(
        "border rounded-md flex gap-2 p-3 flex-wrap",
        "dark:bg-white/20 bg-black/10",
        vertical ? "flex-col grow" : "flex-row"
      )}
    >
      <RunButton
        loadingPlan={loadingPlan}
        run={() =>
          validationMode
            ? runPlan(date!, site!, programs, numNight, semesterVisibility)
            : runPlan(date!, startTime!, endTime!, site!, programs)
        }
        isRunDisabled={isRunDisabled}
      />
      <SiteSelector
        site={site}
        setSite={setSite}
        loadingPlan={loadingPlan}
        vertical={vertical}
        showAll={validationMode}
      />
      <VisibilityRange date={date} setDate={setDate} vertical={vertical} />
      {!validationMode && (
        <DateTimeSelector
          dateTime={startTime!}
          setDateTime={setStartTime}
          setToNow={setToNow}
          setToNowButton={true}
          vertical={vertical}
        />
      )}
      {!validationMode && (
        <DateTimeSelector
          dateTime={endTime!}
          setDateTime={setEndTime}
          setToNow={() => {}}
          setToNowButton={false}
          vertical={vertical}
        />
      )}
      {validationMode && (
        <SemesterVisibility
          semesterVisibility={semesterVisibility}
          setSemesterVisibility={setSemesterVisibility}
        />
      )}
      {validationMode && (
        <NightsNumber
          vertical={vertical}
          nights={numNight}
          setNights={setNumNight}
        />
      )}
      <ProgramSelectorDialog
        programs={programs}
        setProgram={setProgram}
        resetPrograms={() => updatePrograms(structuredClone(programList))}
        validationMode={validationMode}
      />
    </div>
  );
}
