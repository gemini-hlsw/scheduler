import { Button } from "@/components/ui/button";
import { useState } from "react";
import { ButtonGroup } from "@/components/ui/button-group";
import { cn } from "@/lib/utils";
import { type ProgramListType } from "./ProgramList";
import { Field } from "@/components/ui/field";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";

export function StyledButton({
  semester,
  semesterFilter,
  setSemesterFilter,
}: {
  semester: string;
  semesterFilter: string;
  setSemesterFilter: (semester: string) => void;
}) {
  return (
    <Button
      variant="default"
      size="sm"
      className={cn(
        "flex grow",
        "dark:text-white text-black",
        "dark:hover:bg-blue-700 hover:bg-blue-500",
        "px-2",
        semesterFilter === semester
          ? "dark:bg-blue-800 bg-blue-400"
          : "dark:bg-white/20 bg-black/20"
      )}
      onClick={() => setSemesterFilter(semester)}
    >
      {semester}
    </Button>
  );
}

export function ValidationProgramSelectorSite({
  siteName,
  siteLabel,
  programs,
  setProgram,
}: {
  siteName: string;
  siteLabel: string;
  programs: ProgramListType[];
  setProgram: (program: string, state: boolean) => void;
}) {
  const [semesterFilter, setSemesterFilter] = useState("2018B");

  function selectAllFiltered() {
    programs
      .filter(
        (p) => p.id.startsWith(siteLabel) && p.id.includes(semesterFilter)
      )
      .forEach((program) => !program.disabled && setProgram(program.id, true));
  }

  function clearAllFiltered() {
    programs
      .filter(
        (p) => p.id.startsWith(siteLabel) && p.id.includes(semesterFilter)
      )
      .forEach((program) => !program.disabled && setProgram(program.id, false));
  }

  return (
    <div className="flex flex-col gap-2 grow border p-3 rounded-md">
      <h1>{siteName}</h1>
      <ButtonGroup className="w-full">
        <StyledButton
          semester="2018A"
          semesterFilter={semesterFilter}
          setSemesterFilter={setSemesterFilter}
        />
        <StyledButton
          semester="2018B"
          semesterFilter={semesterFilter}
          setSemesterFilter={setSemesterFilter}
        />
        <StyledButton
          semester="2019A"
          semesterFilter={semesterFilter}
          setSemesterFilter={setSemesterFilter}
        />
        <StyledButton
          semester="2019B"
          semesterFilter={semesterFilter}
          setSemesterFilter={setSemesterFilter}
        />
      </ButtonGroup>
      <div className="flex flex-row gap-2 grow">
        <Button
          className={cn(
            "flex grow",
            "dark:bg-blue-800 bg-blue-400",
            "dark:text-white text-black",
            "dark:hover:bg-blue-700 hover:bg-blue-500"
          )}
          onClick={selectAllFiltered}
        >
          Select all filtered
        </Button>
        <Button
          className={cn(
            "flex grow",
            "dark:bg-blue-800 bg-blue-400",
            "dark:text-white text-black",
            "dark:hover:bg-blue-700 hover:bg-blue-500"
          )}
          onClick={clearAllFiltered}
        >
          Clear all filtered
        </Button>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 xl:grid-cols-6 gap-1 h-80 overflow-y-auto">
        {programs
          .filter(
            (p) => p.id.startsWith(siteLabel) && p.id.includes(semesterFilter)
          )
          .map((program) => (
            <Field orientation="horizontal" key={program.id}>
              <Checkbox
                id={program.id}
                name={program.id}
                checked={program.checked}
                disabled={program.disabled}
                onCheckedChange={(checked) =>
                  setProgram(program.id, checked as boolean)
                }
              />
              <Label htmlFor={program.id}>{program.label.substring(9)}</Label>
            </Field>
          ))}
      </div>
    </div>
  );
}
