export function toUtcIsoString(date: Date) {
  const utcDate = new Date(date);
  utcDate.setHours(utcDate.getHours() - utcDate.getTimezoneOffset() / 60);
  return utcDate.toISOString().split(".")[0].replace("T", " ");
}

export function getOffset(timeZone = "UTC", date = new Date()) {
  const utcDate = new Date(date.toLocaleString("en-US", { timeZone: "UTC" }));
  const tzDate = new Date(date.toLocaleString("en-US", { timeZone }));
  return (tzDate.getTime() - utcDate.getTime()) / 6e4;
}

export function getSiteOffset(site: string) {
  return site === "GN"
    ? getOffset("Pacific/Honolulu")
    : getOffset("America/Santiago");
}

export function utcToLocal(date: Date, offset: number) {
  return date.getTime() + offset * 60 * 1000;
}

function getTimezoneOffsetString(timeZone: string, date = new Date()) {
  const dtf = new Intl.DateTimeFormat("en-US", {
    timeZone,
    hour12: false,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  const parts: Intl.DateTimeFormatPart[] = dtf.formatToParts(date);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const map: any = {};
  for (const { type, value } of parts) {
    map[type] = value;
  }

  const asUTC = Date.UTC(
    map.year,
    map.month - 1,
    map.day,
    map.hour,
    map.minute,
    map.second,
  );

  const offsetMinutes = (asUTC - date.getTime()) / 60000;

  const sign = offsetMinutes >= 0 ? "+" : "-";
  const abs = Math.abs(offsetMinutes);
  const hours = String(Math.floor(abs / 60)).padStart(2, "0");
  const minutes = String(Math.round(abs % 60)).padStart(2, "0");

  return `${sign}${hours}:${minutes}`;
}

export function tzDateToString(date: Date, timeZone: string) {
  const offset = getTimezoneOffsetString(timeZone);
  return new Date(toUtcIsoString(date) + offset)
    .toISOString()
    .split(".")[0]
    .replace("T", " ");
}

export function stringDateToLocalString(date: string, site: string) {
  return new Date(utcToLocal(new Date(date), getSiteOffset(site)))
    .toISOString()
    .split(".")[0]
    .replace("T", " ")
    .substring(0, 16);
}
