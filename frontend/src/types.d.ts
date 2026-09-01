export type Theme = "light" | "dark";

export interface NightPlanType {
  nightIndex: number;
  timeEntriesBySite: TimeEntriesBySite[];
}

export interface RtPlanType {
  nightIdx: number;
  plansPerSite: PlanPerSite[];
}

export interface PlanPerSite {
  endTime: string;
  site: string;
  startTime: string;
  visits: Visit[];
  nightConditions: NightConditions;
  nightStats: NightStats;
}

export interface TimeStats {
  nightLength: number;
  observed: number;
  scheduled: number;
  weather: number;
  fault: number;
  closed: number;
  unscheduled: number;
}

export interface TimeEntriesBySite {
  site: string;
  eveTwilight: string;
  mornTwilight: string;
  timeEntries: TimeEntryType[];
}

export interface TimeEntryType {
  startTimeSlots: number;
  event: Event;
  plan: Plan;
  timelossWindows: TimeLossWindow[];
  timestats: TimeStats;
}

export interface TimeLossWindow {
  start: string;
  end?: string | null | undefined;
  lossType: string;
}

export interface Event {
  time: string;
  site: string;
  description: string;
}

export interface Plan {
  startTime: string;
  visits: Visit[];
  nightStats: NightStats;
  nightConditions?: NightConditions;
}

export interface NightConditions {
  cc: string;
  iq: string;
}

export interface Visit {
  obsId: string;
  endTime: string;
  altitude: number[];
  atomEndIdx: number;
  atomStartIdx: number;
  startTime: string;
  instrument: string;
  fpu: string;
  disperser: string;
  filters: string[];
  score: number;
  completion: string;
  atomTimes: number[];
  obsClass: string;
  peakScore: number;
  requiredConditions: NightConditions;
}

export interface NightStats {
  planScore: number;
  nToos: number;
  completionFraction: number[];
  programCompletion: { [key: string]: number };
}

export interface RunSummary {
  summary: { [key: string]: number[] };
  metricsPerBand: { [key: string]: number };
}
