import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { ProgramSelector } from "./ProgramSelection/ProgramSelector";
import { cn } from "@/lib/utils";
import { type ProgramListType } from "./ProgramSelection/ProgramList";
import { FaList } from "react-icons/fa";
import { ImSpinner9 } from "react-icons/im";

export function ProgramSelectorDialog({
  programs,
  setProgram,
  resetPrograms,
  validationMode = false,
  full = false,
  loading = false,
}: {
  programs: ProgramListType[];
  setProgram: (program: string, state: boolean) => void;
  resetPrograms: () => void;
  validationMode?: boolean;
  full?: boolean;
  loading?: boolean;
}) {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button
          disabled={loading}
          variant="default"
          className={cn(
            "dark:bg-blue-800 bg-blue-400",
            "dark:text-white text-black",
            "dark:hover:bg-blue-700 hover:bg-blue-500",
            full ? "w-full" : ""
          )}
        >
          {loading ? <ImSpinner9 className="animate-spin" /> : <FaList />}
          Program Selection
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-11/12 max-h-5/6">
        <DialogTitle className="text-center">Program Selection</DialogTitle>
        <ProgramSelector
          programs={programs}
          setProgram={setProgram}
          resetPrograms={resetPrograms}
          validationMode={validationMode}
        />
      </DialogContent>
    </Dialog>
  );
}
