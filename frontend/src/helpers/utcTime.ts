export function toUtcIsoString(date: Date) {
  const utcDate = new Date(date);
  utcDate.setHours(utcDate.getHours() - utcDate.getTimezoneOffset() / 60);
  return utcDate.toISOString().split(".")[0].replace("T", " ");
}
