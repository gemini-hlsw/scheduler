import { Checkbox } from "@/components/ui/checkbox";
import { Field } from "@/components/ui/field";
import { Label } from "@/components/ui/label";
import { type ProgramListType } from "./ProgramList";

export function OperationProgramSelector({
  programs,
  setProgram,
}: {
  programs: ProgramListType[];
  setProgram: (program: string, state: boolean) => void;
}) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 xl:grid-cols-6 gap-1">
      {programs.map((program) => (
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
          <Label htmlFor={program.id}>{program.label}</Label>
        </Field>
      ))}
    </div>
  );
}
