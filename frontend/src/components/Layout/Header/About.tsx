import { scheduleVersionQuery } from "./query";
import { useQuery } from "@apollo/client";
import {
  Dialog,
  DialogTrigger,
  DialogContent,
  DialogTitle,
} from "@/components/ui/dialog";
import uiVersion from "../../../version.json";
import { cn } from "../../../lib/utils";
import { Button } from "@/components/ui/button";
import { ImSpinner9 } from "react-icons/im";

export function About() {
  const { loading, data: scheduleVersion } = useQuery(scheduleVersionQuery, {
    fetchPolicy: "no-cache",
  });

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="outline" size="xs">
          <span className="label">About</span>
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogTitle className="text-center">About</DialogTitle>
        <div className={cn("flex flex-col md:flex-row gap-6 pb-4")}>
          <div>
            <h4 className="font-bold">
              Server version{" "}
              {loading ? (
                <ImSpinner9 className="animate-spin" />
              ) : (
                scheduleVersion?.version?.version
              )}
            </h4>
          </div>
          <div>
            <h4 className="font-bold">UI version {uiVersion.version}</h4>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
