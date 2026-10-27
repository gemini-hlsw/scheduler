import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { SITE_TIME_ZONES } from "@/helpers/utcTime";

// The current local time at each site, ticking every second.
export function SiteClocks() {
  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <>
      {Object.entries(SITE_TIME_ZONES).map(([site, timeZone]) => (
        <Badge
          key={site}
          variant="outline"
          className="text-xs font-mono"
          title={`${site} local time (${timeZone})`}
        >
          {site}{" "}
          {now.toLocaleTimeString("en-GB", {
            timeZone,
            hourCycle: "h23",
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
          })}
        </Badge>
      ))}
    </>
  );
}
