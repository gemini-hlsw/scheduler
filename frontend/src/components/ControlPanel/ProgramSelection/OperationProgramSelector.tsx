import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Field } from "@/components/ui/field";
import { Label } from "@/components/ui/label";
import { type ProgramListType } from "./ProgramList";

const PAGE_SIZE = 102;

export function OperationProgramSelector({
  programs,
  setProgram,
}: {
  programs: ProgramListType[];
  setProgram: (program: string, state: boolean) => void;
}) {
  const [page, setPage] = useState(0);

  // Sort before paging so the order runs across pages, not just within one.
  const sortedPrograms = [...programs].sort((a, b) =>
    a.label.localeCompare(b.label, undefined, { numeric: true }),
  );
  const pageCount = Math.max(1, Math.ceil(programs.length / PAGE_SIZE));
  // The list can shrink under us (programs reloaded), so never point past the last page.
  const currentPage = Math.min(page, pageCount - 1);
  const pagePrograms = sortedPrograms.slice(
    currentPage * PAGE_SIZE,
    (currentPage + 1) * PAGE_SIZE,
  );

  return (
    <div className="flex flex-col gap-2">
      {/* Multi-column layout, not a grid, so the sorted list runs down each column first. */}
      <div className="columns-2 md:columns-3 lg:columns-5 xl:columns-6 gap-1">
        {pagePrograms.map((program) => (
          <Field
            orientation="horizontal"
            key={program.id}
            // Padding, not margin: a margin would push down the top of every column after the first.
            className="break-inside-avoid pb-1"
          >
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
