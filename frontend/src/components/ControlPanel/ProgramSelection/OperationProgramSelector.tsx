import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Field } from "@/components/ui/field";
import { Label } from "@/components/ui/label";
import { type ProgramListType } from "./ProgramList";

const PAGE_SIZE = 40;

export function OperationProgramSelector({
  programs,
  setProgram,
}: {
  programs: ProgramListType[];
  setProgram: (program: string, state: boolean) => void;
}) {
  const [page, setPage] = useState(0);

  const pageCount = Math.max(1, Math.ceil(programs.length / PAGE_SIZE));
  // The list can shrink under us (programs reloaded), so never point past the last page.
  const currentPage = Math.min(page, pageCount - 1);
  const pagePrograms = programs.slice(
    currentPage * PAGE_SIZE,
    (currentPage + 1) * PAGE_SIZE
  );

  return (
    <div className="flex flex-col gap-2">
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 xl:grid-cols-6 gap-1">
        {pagePrograms.map((program) => (
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
      <div className="flex flex-row items-center justify-center gap-2">
        <Button
          variant="outline"
          size="sm"
          disabled={currentPage === 0}
          onClick={() => setPage(currentPage - 1)}
        >
          Previous
        </Button>
        <span className="text-sm tabular-nums">
          Page {currentPage + 1} of {pageCount} ({programs.length} programs)
        </span>
        <Button
          variant="outline"
          size="sm"
          disabled={currentPage >= pageCount - 1}
          onClick={() => setPage(currentPage + 1)}
        >
          Next
        </Button>
      </div>
    </div>
  );
}
