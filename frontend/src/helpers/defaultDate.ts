export function getDefaultDate(validationMode: boolean = false) {
  const defaultDate: Date = validationMode
    ? new Date("2018-10-21")
    : new Date();
  defaultDate.setDate(defaultDate.getDate() + 1);
  defaultDate.setHours(0);
  defaultDate.setMinutes(0);
  defaultDate.setSeconds(0);
  return defaultDate;
}
