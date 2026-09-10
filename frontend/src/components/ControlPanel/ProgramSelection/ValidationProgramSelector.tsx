import { type ProgramListType } from "./ProgramList";
import { ValidationProgramSelectorSite } from "./ValidationProgramSelectorSite";

export function ValidationProgramSelector({
  programs,
  setProgram,
}: {
  programs: ProgramListType[];
  setProgram: (program: string, state: boolean) => void;
}) {
  return (
    <div className="flex flex-row gap-2">
      <ValidationProgramSelectorSite
        siteLabel="GN"
        siteName="Gemini North"
        programs={programs}
        setProgram={setProgram}
      />
      <ValidationProgramSelectorSite
        siteLabel="GS"
        siteName="Gemini South"
        programs={programs}
        setProgram={setProgram}
      />
    </div>
  );
}
