import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useState } from "react";
import { Field } from "@/components/ui/field";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import { type ProgramListType } from "./ProgramList";
import { ValidationProgramSelector } from "./ValidationProgramSelector";
import { OperationProgramSelector } from "./OperationProgramSelector";

export function ProgramSelector({
  programs,
  setProgram,
  resetPrograms,
  validationMode = false,
}: {
  programs: ProgramListType[];
  setProgram: (program: string, state: boolean) => void;
  resetPrograms: () => void;
  validationMode?: boolean;
}) {
  const [filter, setFilter] = useState("");

  async function fileUpload(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files[0];
    const reader = new FileReader();
    // let blob = await fetch(file.objectURL).then((r) => r.blob());
    reader.readAsText(file);

    reader.onloadend = function () {
      const text = reader.result;
      const list = (text as string).split("\n").filter((e) => e);

      programs.forEach((program) => {
        setProgram(
          program.id,
          program.disabled
            ? program.checked
            : list.includes(program.id)
            ? true
            : false
        );
      });
    };
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-row gap-2 grow">
        <Button
          variant="default"
          className={cn(
            "grow",
            "dark:bg-blue-800 bg-blue-400",
            "dark:text-white text-black",
            "dark:hover:bg-blue-700 hover:bg-blue-500"
          )}
          onClick={() => {
            programs
              .filter((p) => !p.disabled)
              .forEach((program) => {
                setProgram(program.id, true);
              });
          }}
        >
          Select All
        </Button>
        <Button
          variant="default"
          className={cn(
            "grow",
            "dark:bg-blue-800 bg-blue-400",
            "dark:text-white text-black",
            "dark:hover:bg-blue-700 hover:bg-blue-500"
          )}
          onClick={() => {
            programs
              .filter((p) => !p.disabled)
              .forEach((program) => {
                setProgram(program.id, false);
              });
          }}
        >
          Clear All
        </Button>
        <Button
          variant="default"
          className={cn(
            "grow",
            "dark:bg-blue-800 bg-blue-400",
            "dark:text-white text-black",
            "dark:hover:bg-blue-700 hover:bg-blue-500"
          )}
          onClick={resetPrograms}
        >
          Reset to default
        </Button>
        {validationMode && (
          <Field orientation="horizontal" className="w-1/4">
            <Label
              htmlFor="file-upload"
              className={cn(
                "p-3 rounded-md cursor-pointer",
                "dark:bg-blue-800 bg-blue-400",
                "dark:text-white text-black",
                "dark:hover:bg-blue-700 hover:bg-blue-500"
              )}
            >
              Load from file
            </Label>
            <Input
              id="file-upload"
              type="file"
              className={"hidden"}
              onChange={fileUpload}
            />
          </Field>
        )}
      </div>
      <Input
        type="text"
        placeholder="Filter Programs"
        onChange={(e) => setFilter(e.target.value)}
      />
      {validationMode ? (
        <ValidationProgramSelector
          programs={programs.filter((p) =>
            p.label.toLowerCase().includes(filter.toLowerCase())
          )}
          setProgram={setProgram}
        />
      ) : (
        <OperationProgramSelector
          programs={programs.filter((p) =>
            p.label.toLowerCase().includes(filter.toLowerCase())
          )}
          setProgram={setProgram}
        />
      )}
    </div>
  );
}
