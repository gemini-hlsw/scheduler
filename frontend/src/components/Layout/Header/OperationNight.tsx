import { useContext } from "react";
import { useMatch } from "react-router-dom";
import { GlobalStateContext } from "../../GlobalState/GlobalState";
import { Badge } from "@/components/ui/badge";
import { stringDateToLocalString } from "@/helpers/utcTime";

// The default night the operation server plans for, per site, in site local
// time: the 12-degree twilights, whatever custom night times a build sets.
// Nothing from servers that are not in operation mode, and shown only on the
// Operation tab.
export function OperationNight() {
  const { operationNight } = useContext(GlobalStateContext);
  const onOperationTab = useMatch("/operation") !== null;

  if (!onOperationTab) return null;

  return (
    <>
      {operationNight.map(({ site, start, end }) => {
        const localStart = stringDateToLocalString(start, site);
        const localEnd = stringDateToLocalString(end, site);
        // The start date names the night; its end is the next morning, so the
        // badge shows only the time and the hover text has both full dates.
        return (
          <Badge
            key={site}
            variant="outline"
            className="text-xs font-mono"
            title={`${site} default night (12° twilights): ${localStart} to ${localEnd} local, ${start} to ${end} UTC`}
          >
            {site} {localStart} – {localEnd.substring(11)}
          </Badge>
        );
      })}
    </>
  );
}
