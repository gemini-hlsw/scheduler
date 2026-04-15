import { useQuery, useSubscription } from "@apollo/client";
import { buildParametersQuery } from "./query";
import { buildParametersSubscription } from "./subscription";
import {
  BuildParametersQuery,
  BuildParametersUpdatesSubscription,
} from "@/gql/graphql";
import {
  getSiteOffset,
  stringDateToLocalString,
  utcToLocal,
} from "@/helpers/utcTime";
import { cn } from "@/lib/utils";

function getLatestData(
  queryData: BuildParametersQuery,
  subscriptionData: BuildParametersUpdatesSubscription,
) {
  if (subscriptionData) {
    const gnStart = subscriptionData.buildParametersUpdates?.nightTimes?.find(
      (x) => x.site === "Gemini North",
    )?.start;
    const gnEnd = subscriptionData.buildParametersUpdates?.nightTimes?.find(
      (x) => x.site === "Gemini North",
    )?.end;
    const gsStart = subscriptionData.buildParametersUpdates?.nightTimes?.find(
      (x) => x.site === "Gemini South",
    )?.start;
    const gsEnd = subscriptionData.buildParametersUpdates?.nightTimes?.find(
      (x) => x.site === "Gemini South",
    )?.end;

    return {
      visibilityStart:
        subscriptionData.buildParametersUpdates?.visibilityStart ?? "Default",
      visibilityEnd:
        subscriptionData.buildParametersUpdates?.visibilityEnd ?? "Default",
      gnStart: gnStart
        ? stringDateToLocalString(gnStart + "Z", "GN")
        : "Default",
      gnEnd: gnEnd ? stringDateToLocalString(gnEnd + "Z", "GN") : "Default",
      gsStart: gsStart
        ? stringDateToLocalString(gsStart + "Z", "GS")
        : "Default",
      gsEnd: gsEnd ? stringDateToLocalString(gsEnd + "Z", "GS") : "Default",
    };
  } else {
    if (queryData) {
      const gnStart = queryData.buildParameters?.nightTimes?.find(
        (x) => x.site === "Gemini North",
      )?.start;
      const gnEnd = queryData.buildParameters?.nightTimes?.find(
        (x) => x.site === "Gemini North",
      )?.end;
      const gsStart = queryData.buildParameters?.nightTimes?.find(
        (x) => x.site === "Gemini South",
      )?.start;
      const gsEnd = queryData.buildParameters?.nightTimes?.find(
        (x) => x.site === "Gemini South",
      )?.end;

      return {
        visibilityStart:
          queryData.buildParameters?.visibilityStart ?? "Default",
        visibilityEnd: queryData.buildParameters?.visibilityEnd ?? "Default",
        gnStart: gnStart
          ? stringDateToLocalString(gnStart + "Z", "GN")
          : "Default",
        gnEnd: gnEnd ? stringDateToLocalString(gnEnd + "Z", "GN") : "Default",
        gsStart: gsStart
          ? stringDateToLocalString(gsStart + "Z", "GS")
          : "Default",
        gsEnd: gsEnd ? stringDateToLocalString(gsEnd + "Z", "GS") : "Default",
      };
    } else {
      return {
        visibilityStart: "Default",
        visibilityEnd: "Default",
        gnStart: "Default",
        gnEnd: "Default",
        gsStart: "Default",
        gsEnd: "Default",
      };
    }
  }
}

function TimeBadge({
  time,
  dateOnly = false,
}: {
  time: string;
  dateOnly?: boolean;
}) {
  if (time === "Default") {
    return (
      <span
        className={cn(
          "text-red-700 dark:text-red-200",
          "py-1 px-2 rounded-full text-sm font-mono",
        )}
      >
        Default
      </span>
    );
  }

  return (
    <span
      className={cn(
        "text-blue-700 dark:text-blue-100",
        "py-1 px-2 rounded-full text-sm font-mono",
      )}
    >
      {dateOnly
        ? time.substring(0, 10)
        : time.replace("T", " ").substring(0, 16)}
    </span>
  );
}

export default function DisplayBuildParameters() {
  const { data: queryData } = useQuery(buildParametersQuery, {
    fetchPolicy: "no-cache",
    context: { clientName: "realtimeClient" },
  });

  const { data: subscriptionData } = useSubscription(
    buildParametersSubscription,
  );

  const data = getLatestData(queryData, subscriptionData);

  return (
    <div className="flex flex-col gap-1">
      <h1 className="font-bold text-nowrap px-2">Last Updated</h1>
      <div className="h-9 text-nowrap flex flex-row gap-1 items-center">
        <TimeBadge time={data.visibilityStart} dateOnly={true} />
        -
        <TimeBadge time={data.visibilityEnd} dateOnly={true} />
      </div>
      <div className="h-9 text-nowrap flex flex-row items-center">
        <TimeBadge time={data.gnStart} />
      </div>
      <div className="h-9 text-nowrap flex flex-row items-center">
        <TimeBadge time={data.gnEnd} />
      </div>
      <div className="h-9 text-nowrap flex flex-row items-center">
        <TimeBadge time={data.gsStart} />
      </div>
      <div className="h-9 text-nowrap flex flex-row items-center">
        <TimeBadge time={data.gsEnd} />
      </div>
    </div>
  );
}
