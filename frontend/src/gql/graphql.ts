/* eslint-disable */
import { TypedDocumentNode as DocumentNode } from '@graphql-typed-document-node/core';
export type Maybe<T> = T | null;
export type InputMaybe<T> = Maybe<T>;
export type Exact<T extends { [key: string]: unknown }> = { [K in keyof T]: T[K] };
export type MakeOptional<T, K extends keyof T> = Omit<T, K> & { [SubKey in K]?: Maybe<T[SubKey]> };
export type MakeMaybe<T, K extends keyof T> = Omit<T, K> & { [SubKey in K]: Maybe<T[SubKey]> };
/** All built-in and custom scalars, mapped to their actual values */
export type Scalars = {
  ID: string;
  String: string;
  Boolean: boolean;
  Int: number;
  Float: number;
  /** Date (isoformat) */
  Date: any;
  /** Date with time (isoformat) */
  DateTime: any;
  /** The `JSON` scalar type represents JSON values as specified by [ECMA-404](https://ecma-international.org/wp-content/uploads/ECMA-404_2nd_edition_december_2017.pdf). */
  JSON: any;
  /** ID of an Observation */
  SObservationID: any;
  /** Depiction of the sites that can be load to the collector */
  Sites: any;
};

export type AvailableProgram = {
  __typename?: 'AvailableProgram';
  id: Scalars['String'];
  refLabel: Scalars['String'];
};

export type BuildParametersInput = {
  nightTimes?: InputMaybe<Array<SiteNightTimesEntry>>;
  programList?: InputMaybe<Array<Scalars['String']>>;
  visibilityEnd?: InputMaybe<Scalars['DateTime']>;
  visibilityStart?: InputMaybe<Scalars['DateTime']>;
};

export type BuildParametersResponse = {
  __typename?: 'BuildParametersResponse';
  nightTimes?: Maybe<Array<NightTimesResponse>>;
  programList?: Maybe<Array<Scalars['String']>>;
  visibilityEnd?: Maybe<Scalars['DateTime']>;
  visibilityStart?: Maybe<Scalars['DateTime']>;
};

export type CreateNewScheduleInput = {
  airPower?: InputMaybe<Scalars['Float']>;
  endTime: Scalars['String'];
  metPower?: InputMaybe<Scalars['Float']>;
  mode: SchedulerModes;
  numNightsToSchedule?: InputMaybe<Scalars['Int']>;
  power?: InputMaybe<Scalars['Int']>;
  programs?: InputMaybe<Array<Scalars['String']>>;
  semesterVisibility?: Scalars['Boolean'];
  sites: Scalars['Sites'];
  startTime: Scalars['String'];
  thesisFactor?: InputMaybe<Scalars['Float']>;
  visPower?: InputMaybe<Scalars['Float']>;
  whaPower?: InputMaybe<Scalars['Float']>;
};

export type GroupCoverage = {
  __typename?: 'GroupCoverage';
  expected: Scalars['Int'];
  key: Scalars['String'];
  missing: Scalars['Int'];
  pending: Scalars['Int'];
  skipped: Scalars['Int'];
  stored: Scalars['Int'];
};

export type Event = {
  __typename?: 'Event';
  description: Scalars['String'];
  site: Site;
  time: Scalars['DateTime'];
};

export type Mutation = {
  __typename?: 'Mutation';
  updateBuildParams: Scalars['String'];
  updateWeather?: Maybe<Weather>;
};


export type MutationUpdateBuildParamsArgs = {
  buildParamsInput: BuildParametersInput;
};


export type MutationUpdateWeatherArgs = {
  weatherInput?: InputMaybe<WeatherInput>;
};

export type NewNightPlans = {
  __typename?: 'NewNightPlans';
  nightPlans: SNightTimelines;
  plansSummary: SRunSummary;
};

export type NewPlansRt = {
  __typename?: 'NewPlansRT';
  nightPlans: SPlans;
};

export type NightPlansError = {
  __typename?: 'NightPlansError';
  error: Scalars['String'];
};

export type NightPlansResponseRt = NewNightPlans | NewPlansRt | NightPlansError | NightPlansWithEvent;

export type NightPlansWithEvent = {
  __typename?: 'NightPlansWithEvent';
  event: Scalars['String'];
  nightPlans: SPlans;
};

export type NightTimesInput = {
  nightEnd?: InputMaybe<Scalars['DateTime']>;
  nightStart?: InputMaybe<Scalars['DateTime']>;
};

export type NightTimesResponse = {
  __typename?: 'NightTimesResponse';
  end?: Maybe<Scalars['DateTime']>;
  site: Scalars['String'];
  start?: Maybe<Scalars['DateTime']>;
};

export type ObservationCoverage = {
  __typename?: 'ObservationCoverage';
  observationId: Scalars['String'];
  programLabel: Scalars['String'];
  reason?: Maybe<Scalars['String']>;
  site?: Maybe<Scalars['String']>;
  status: ObservationStatus;
  targetName?: Maybe<Scalars['String']>;
};

export type ObservationCoveragePage = {
  __typename?: 'ObservationCoveragePage';
  nightDate?: Maybe<Scalars['Date']>;
  observations: Array<ObservationCoverage>;
  odbReadAt?: Maybe<Scalars['DateTime']>;
  total: Scalars['Int'];
};

export type ObservationStatus =
  | 'MISSING'
  | 'PENDING'
  | 'SKIPPED'
  | 'STORED';

export type Query = {
  __typename?: 'Query';
  availablePrograms: Array<AvailableProgram>;
  buildParameters: BuildParametersResponse;
  observationCoverage: ObservationCoveragePage;
  onDemandSchedule: Scalars['String'];
  schedule: Scalars['String'];
  scheduleV2: Scalars['String'];
  version: Version;
  visibilityAggregatorStatus: VisibilityAggregatorStatus;
  visibilityCoverage: VisibilityCoverage;
  visibleObservations: VisibleObservationsPage;
  weather?: Maybe<Array<Maybe<Weather>>>;
};


export type QueryObservationCoverageArgs = {
  limit?: Scalars['Int'];
  nightDate?: InputMaybe<Scalars['Date']>;
  offset?: Scalars['Int'];
  programLabel?: InputMaybe<Scalars['String']>;
  search?: InputMaybe<Scalars['String']>;
  site?: InputMaybe<Scalars['String']>;
  status?: InputMaybe<ObservationStatus>;
};


export type QueryScheduleArgs = {
  newScheduleInput: CreateNewScheduleInput;
  scheduleId: Scalars['String'];
};


export type QueryVisibilityCoverageArgs = {
  nightDate?: InputMaybe<Scalars['Date']>;
};


export type QueryVisibleObservationsArgs = {
  limit?: Scalars['Int'];
  minRemainingMinutes?: Scalars['Int'];
  nightDate?: InputMaybe<Scalars['Date']>;
  offset?: Scalars['Int'];
  site: Scalars['String'];
};

export type SConditions = {
  __typename?: 'SConditions';
  cc: Scalars['String'];
  iq: Scalars['String'];
};

export type SNightInTimeline = {
  __typename?: 'SNightInTimeline';
  nightIndex: Scalars['Int'];
  timeEntriesBySite: Array<TimelineEntriesBySite>;
};

export type SNightStats = {
  __typename?: 'SNightStats';
  completionFraction: Scalars['JSON'];
  nToos: Scalars['Int'];
  planScore: Scalars['Float'];
  programCompletion: Scalars['JSON'];
  timeLoss: Scalars['JSON'];
};

export type SNightTimelines = {
  __typename?: 'SNightTimelines';
  nightTimeline: Array<SNightInTimeline>;
};

export type SPlan = {
  __typename?: 'SPlan';
  endTime: Scalars['DateTime'];
  nightConditions: SConditions;
  nightStats: SNightStats;
  site: Site;
  startTime: Scalars['DateTime'];
  visits: Array<SVisit>;
};

export type SPlans = {
  __typename?: 'SPlans';
  nightIdx: Scalars['Int'];
  plansPerSite: Array<SPlan>;
};

export type SRunSummary = {
  __typename?: 'SRunSummary';
  metricsPerBand: Scalars['JSON'];
  summary: Scalars['JSON'];
};

export type STimeLossWindow = {
  __typename?: 'STimeLossWindow';
  end?: Maybe<Scalars['DateTime']>;
  lossType: Scalars['String'];
  start: Scalars['DateTime'];
};

export type STimelineEntry = {
  __typename?: 'STimelineEntry';
  event: Event;
  plan: SPlan;
  startTimeSlots: Scalars['Int'];
  timelossWindows: Array<STimeLossWindow>;
};

export type SVisit = {
  __typename?: 'SVisit';
  altitude: Array<Scalars['Float']>;
  atomEndIdx: Scalars['Int'];
  atomStartIdx: Scalars['Int'];
  atomTimes: Array<Scalars['Int']>;
  completion: Scalars['String'];
  disperser: Scalars['String'];
  endTime: Scalars['DateTime'];
  filters: Array<Scalars['String']>;
  fpu: Scalars['String'];
  instrument: Scalars['String'];
  obsClass: Scalars['String'];
  obsId: Scalars['SObservationID'];
  peakScore: Scalars['Float'];
  requiredConditions: SConditions;
  score: Scalars['Float'];
  startTime: Scalars['DateTime'];
};

export type SchedulerModes =
  | 'OPERATION'
  | 'SIMULATION'
  | 'VALIDATION';

export type Site =
  | 'GN'
  | 'GS';

export type SiteNightTimesEntry = {
  nightTimes: NightTimesInput;
  site: Site;
};

export type Subscription = {
  __typename?: 'Subscription';
  buildParametersUpdates: BuildParametersResponse;
  queueSchedule: NightPlansResponseRt;
  weatherUpdates?: Maybe<Weather>;
};


export type SubscriptionQueueScheduleArgs = {
  scheduleId: Scalars['String'];
};

export type TimelineEntriesBySite = {
  __typename?: 'TimelineEntriesBySite';
  eveTwilight: Scalars['DateTime'];
  mornTwilight: Scalars['DateTime'];
  site: Site;
  timeEntries: Array<STimelineEntry>;
  timeLosses: Scalars['JSON'];
};

export type Version = {
  __typename?: 'Version';
  changelog: Array<Scalars['String']>;
  version: Scalars['String'];
};

export type VisibilityAggregatorStatus = {
  __typename?: 'VisibilityAggregatorStatus';
  active: Scalars['Boolean'];
  detail?: Maybe<Scalars['String']>;
  elapsedSeconds?: Maybe<Scalars['Float']>;
  etaSeconds?: Maybe<Scalars['Float']>;
  finishedAt?: Maybe<Scalars['String']>;
  heartbeatAt?: Maybe<Scalars['String']>;
  holder?: Maybe<Scalars['String']>;
  phase?: Maybe<Scalars['String']>;
  progressCurrent?: Maybe<Scalars['Int']>;
  progressTotal?: Maybe<Scalars['Int']>;
  progressUnit?: Maybe<Scalars['String']>;
  stale: Scalars['Boolean'];
  startedAt?: Maybe<Scalars['String']>;
};

export type VisibilityCoverage = {
  __typename?: 'VisibilityCoverage';
  expected: Scalars['Int'];
  isComplete: Scalars['Boolean'];
  missing: Scalars['Int'];
  nightDate?: Maybe<Scalars['Date']>;
  odbReadAt?: Maybe<Scalars['DateTime']>;
  pending: Scalars['Int'];
  pendingKnown: Scalars['Boolean'];
  perProgram: Array<GroupCoverage>;
  perSite: Array<GroupCoverage>;
  skipped: Scalars['Int'];
  stored: Scalars['Int'];
};

export type VisibleInterval = {
  __typename?: 'VisibleInterval';
  end: Scalars['DateTime'];
  start: Scalars['DateTime'];
};

export type VisibleObservation = {
  __typename?: 'VisibleObservation';
  intervals: Array<VisibleInterval>;
  nightDate: Scalars['Date'];
  observationId: Scalars['String'];
  remainingMinutes: Scalars['Int'];
  remainingMinutesFromNow: Scalars['Int'];
  site: Scalars['String'];
  targetName?: Maybe<Scalars['String']>;
};

export type VisibleObservationsPage = {
  __typename?: 'VisibleObservationsPage';
  nightDate: Scalars['Date'];
  observations: Array<VisibleObservation>;
  site: Scalars['String'];
  total: Scalars['Int'];
  totalRemainingMinutes: Scalars['Int'];
};

export type Weather = {
  __typename?: 'Weather';
  cloudCover?: Maybe<Scalars['Float']>;
  imageQuality?: Maybe<Scalars['Float']>;
  site?: Maybe<Scalars['String']>;
  windDirection?: Maybe<Scalars['Float']>;
  windSpeed?: Maybe<Scalars['Float']>;
};

export type WeatherInput = {
  cloudCover?: InputMaybe<Scalars['Float']>;
  imageQuality?: InputMaybe<Scalars['Float']>;
  site?: InputMaybe<Scalars['String']>;
  windDirection?: InputMaybe<Scalars['Float']>;
  windSpeed?: InputMaybe<Scalars['Float']>;
};

export type UpdateBuildParamsMutationVariables = Exact<{
  buildParamsInput: BuildParametersInput;
}>;


export type UpdateBuildParamsMutation = { __typename?: 'Mutation', updateBuildParams: string };

export type AvailableProgramsQueryVariables = Exact<{ [key: string]: never; }>;


export type AvailableProgramsQuery = { __typename?: 'Query', availablePrograms: Array<{ __typename?: 'AvailableProgram', id: string, refLabel: string }> };

export type BuildParametersQueryVariables = Exact<{ [key: string]: never; }>;


export type BuildParametersQuery = { __typename?: 'Query', buildParameters: { __typename?: 'BuildParametersResponse', visibilityStart?: any | null, visibilityEnd?: any | null, programList?: Array<string> | null, nightTimes?: Array<{ __typename?: 'NightTimesResponse', site: string, start?: any | null, end?: any | null }> | null } };

export type BuildParametersUpdatesSubscriptionVariables = Exact<{ [key: string]: never; }>;


export type BuildParametersUpdatesSubscription = { __typename?: 'Subscription', buildParametersUpdates: { __typename?: 'BuildParametersResponse', visibilityStart?: any | null, visibilityEnd?: any | null, programList?: Array<string> | null, nightTimes?: Array<{ __typename?: 'NightTimesResponse', site: string, start?: any | null, end?: any | null }> | null } };

export type QueueScheduleSubscriptionVariables = Exact<{
  scheduleId: Scalars['String'];
}>;


export type QueueScheduleSubscription = { __typename?: 'Subscription', queueSchedule: { __typename: 'NewNightPlans', nightPlans: { __typename?: 'SNightTimelines', nightTimeline: Array<{ __typename?: 'SNightInTimeline', nightIndex: number, timeEntriesBySite: Array<{ __typename?: 'TimelineEntriesBySite', site: Site, mornTwilight: any, eveTwilight: any, timeLosses: any, timeEntries: Array<{ __typename?: 'STimelineEntry', startTimeSlots: number, event: { __typename?: 'Event', time: any, site: Site, description: string }, plan: { __typename?: 'SPlan', startTime: any, nightConditions: { __typename?: 'SConditions', iq: string, cc: string }, visits: Array<{ __typename?: 'SVisit', obsId: any, endTime: any, altitude: Array<number>, atomEndIdx: number, atomStartIdx: number, startTime: any, instrument: string, fpu: string, disperser: string, filters: Array<string>, score: number, obsClass: string, completion: string, atomTimes: Array<number>, peakScore: number, requiredConditions: { __typename?: 'SConditions', iq: string, cc: string } }>, nightStats: { __typename?: 'SNightStats', timeLoss: any, planScore: number, nToos: number, completionFraction: any, programCompletion: any } }, timelossWindows: Array<{ __typename?: 'STimeLossWindow', start: any, end?: any | null, lossType: string }> }> }> }> }, plansSummary: { __typename?: 'SRunSummary', summary: any, metricsPerBand: any } } | { __typename: 'NewPlansRT', nightPlans: { __typename?: 'SPlans', nightIdx: number, plansPerSite: Array<{ __typename?: 'SPlan', endTime: any, site: Site, startTime: any, visits: Array<{ __typename?: 'SVisit', altitude: Array<number>, atomEndIdx: number, atomStartIdx: number, completion: string, disperser: string, endTime: any, filters: Array<string>, fpu: string, instrument: string, obsClass: string, obsId: any, peakScore: number, score: number, startTime: any, requiredConditions: { __typename?: 'SConditions', cc: string, iq: string } }>, nightConditions: { __typename?: 'SConditions', cc: string, iq: string }, nightStats: { __typename?: 'SNightStats', completionFraction: any, nToos: number, planScore: number, programCompletion: any, timeLoss: any } }> } } | { __typename: 'NightPlansError', error: string } | { __typename: 'NightPlansWithEvent', event: string, nightPlans: { __typename?: 'SPlans', nightIdx: number, plansPerSite: Array<{ __typename?: 'SPlan', endTime: any, site: Site, startTime: any, visits: Array<{ __typename?: 'SVisit', altitude: Array<number>, atomEndIdx: number, atomStartIdx: number, completion: string, disperser: string, endTime: any, filters: Array<string>, fpu: string, instrument: string, obsClass: string, obsId: any, peakScore: number, score: number, startTime: any, requiredConditions: { __typename?: 'SConditions', cc: string, iq: string } }>, nightConditions: { __typename?: 'SConditions', cc: string, iq: string }, nightStats: { __typename?: 'SNightStats', completionFraction: any, nToos: number, planScore: number, programCompletion: any, timeLoss: any } }> } } };

export type VersionQueryVariables = Exact<{ [key: string]: never; }>;


export type VersionQuery = { __typename?: 'Query', version: { __typename?: 'Version', version: string, changelog: Array<string> } };

export type ScheduleV2QueryVariables = Exact<{ [key: string]: never; }>;


export type ScheduleV2Query = { __typename?: 'Query', scheduleV2: string };

export type OnDemandQueryQueryVariables = Exact<{ [key: string]: never; }>;


export type OnDemandQueryQuery = { __typename?: 'Query', onDemandSchedule: string };

export type ScheduleQueryVariables = Exact<{
  scheduleId: Scalars['String'];
  startTime: Scalars['String'];
  endTime: Scalars['String'];
  sites: Scalars['Sites'];
  mode: SchedulerModes;
  numNightsToSchedule: Scalars['Int'];
  semesterVisibility: Scalars['Boolean'];
  thesisFactor?: InputMaybe<Scalars['Float']>;
  power?: InputMaybe<Scalars['Int']>;
  metPower?: InputMaybe<Scalars['Float']>;
  whaPower?: InputMaybe<Scalars['Float']>;
  airPower?: InputMaybe<Scalars['Float']>;
  visPower?: InputMaybe<Scalars['Float']>;
  programs: Array<Scalars['String']> | Scalars['String'];
}>;


export type ScheduleQuery = { __typename?: 'Query', schedule: string };

export type VisibilityAggregatorStatusQueryVariables = Exact<{ [key: string]: never; }>;


export type VisibilityAggregatorStatusQuery = { __typename?: 'Query', visibilityAggregatorStatus: { __typename?: 'VisibilityAggregatorStatus', active: boolean, stale: boolean, holder?: string | null, startedAt?: string | null, heartbeatAt?: string | null, finishedAt?: string | null, phase?: string | null, progressCurrent?: number | null, progressTotal?: number | null, progressUnit?: string | null, elapsedSeconds?: number | null, etaSeconds?: number | null } };

export type VisibilityCoverageQueryVariables = Exact<{
  nightDate?: InputMaybe<Scalars['Date']>;
}>;


export type VisibilityCoverageQuery = { __typename?: 'Query', visibilityCoverage: { __typename?: 'VisibilityCoverage', nightDate?: any | null, odbReadAt?: any | null, expected: number, stored: number, pending: number, missing: number, skipped: number, isComplete: boolean, pendingKnown: boolean, perProgram: Array<{ __typename?: 'GroupCoverage', key: string, expected: number, stored: number, pending: number, missing: number, skipped: number }>, perSite: Array<{ __typename?: 'GroupCoverage', key: string, expected: number, stored: number, pending: number, missing: number, skipped: number }> } };

export type ObservationCoverageQueryVariables = Exact<{
  nightDate?: InputMaybe<Scalars['Date']>;
  status?: InputMaybe<ObservationStatus>;
  site?: InputMaybe<Scalars['String']>;
  programLabel?: InputMaybe<Scalars['String']>;
  search?: InputMaybe<Scalars['String']>;
  limit: Scalars['Int'];
  offset: Scalars['Int'];
}>;


export type ObservationCoverageQuery = { __typename?: 'Query', observationCoverage: { __typename?: 'ObservationCoveragePage', total: number, nightDate?: any | null, odbReadAt?: any | null, observations: Array<{ __typename?: 'ObservationCoverage', observationId: string, programLabel: string, site?: string | null, targetName?: string | null, status: ObservationStatus, reason?: string | null }> } };

export type VisibleObservationsQueryVariables = Exact<{
  site: Scalars['String'];
  nightDate?: InputMaybe<Scalars['Date']>;
  limit: Scalars['Int'];
  offset: Scalars['Int'];
  minRemainingMinutes: Scalars['Int'];
}>;


export type VisibleObservationsQuery = { __typename?: 'Query', visibleObservations: { __typename?: 'VisibleObservationsPage', site: string, nightDate: any, total: number, totalRemainingMinutes: number, observations: Array<{ __typename?: 'VisibleObservation', observationId: string, targetName?: string | null, remainingMinutes: number, remainingMinutesFromNow: number, intervals: Array<{ __typename?: 'VisibleInterval', start: any, end: any }> }> } };

export type UpdateWeatherMutationVariables = Exact<{
  weatherInput?: InputMaybe<WeatherInput>;
}>;


export type UpdateWeatherMutation = { __typename?: 'Mutation', updateWeather?: { __typename?: 'Weather', site?: string | null, imageQuality?: number | null, cloudCover?: number | null, windDirection?: number | null, windSpeed?: number | null } | null };

export type GetWeatherQueryVariables = Exact<{ [key: string]: never; }>;


export type GetWeatherQuery = { __typename?: 'Query', weather?: Array<{ __typename?: 'Weather', site?: string | null, imageQuality?: number | null, cloudCover?: number | null, windDirection?: number | null, windSpeed?: number | null } | null> | null };

export type WeatherUpdatesSubscriptionVariables = Exact<{ [key: string]: never; }>;


export type WeatherUpdatesSubscription = { __typename?: 'Subscription', weatherUpdates?: { __typename?: 'Weather', site?: string | null, imageQuality?: number | null, cloudCover?: number | null, windDirection?: number | null, windSpeed?: number | null } | null };


export const UpdateBuildParamsDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"mutation","name":{"kind":"Name","value":"updateBuildParams"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"buildParamsInput"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"BuildParametersInput"}}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"updateBuildParams"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"buildParamsInput"},"value":{"kind":"Variable","name":{"kind":"Name","value":"buildParamsInput"}}}]}]}}]} as unknown as DocumentNode<UpdateBuildParamsMutation, UpdateBuildParamsMutationVariables>;
export const AvailableProgramsDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"availablePrograms"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"availablePrograms"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"id"}},{"kind":"Field","name":{"kind":"Name","value":"refLabel"}}]}}]}}]} as unknown as DocumentNode<AvailableProgramsQuery, AvailableProgramsQueryVariables>;
export const BuildParametersDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"buildParameters"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"buildParameters"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"nightTimes"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"site"}},{"kind":"Field","name":{"kind":"Name","value":"start"}},{"kind":"Field","name":{"kind":"Name","value":"end"}}]}},{"kind":"Field","name":{"kind":"Name","value":"visibilityStart"}},{"kind":"Field","name":{"kind":"Name","value":"visibilityEnd"}},{"kind":"Field","name":{"kind":"Name","value":"programList"}}]}}]}}]} as unknown as DocumentNode<BuildParametersQuery, BuildParametersQueryVariables>;
export const BuildParametersUpdatesDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"subscription","name":{"kind":"Name","value":"buildParametersUpdates"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"buildParametersUpdates"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"nightTimes"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"site"}},{"kind":"Field","name":{"kind":"Name","value":"start"}},{"kind":"Field","name":{"kind":"Name","value":"end"}}]}},{"kind":"Field","name":{"kind":"Name","value":"visibilityStart"}},{"kind":"Field","name":{"kind":"Name","value":"visibilityEnd"}},{"kind":"Field","name":{"kind":"Name","value":"programList"}}]}}]}}]} as unknown as DocumentNode<BuildParametersUpdatesSubscription, BuildParametersUpdatesSubscriptionVariables>;
export const QueueScheduleDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"subscription","name":{"kind":"Name","value":"queueSchedule"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"scheduleId"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"queueSchedule"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"scheduleId"},"value":{"kind":"Variable","name":{"kind":"Name","value":"scheduleId"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"__typename"}},{"kind":"InlineFragment","typeCondition":{"kind":"NamedType","name":{"kind":"Name","value":"NewNightPlans"}},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"nightPlans"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"nightTimeline"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"nightIndex"}},{"kind":"Field","name":{"kind":"Name","value":"timeEntriesBySite"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"site"}},{"kind":"Field","name":{"kind":"Name","value":"mornTwilight"}},{"kind":"Field","name":{"kind":"Name","value":"eveTwilight"}},{"kind":"Field","name":{"kind":"Name","value":"timeLosses"}},{"kind":"Field","name":{"kind":"Name","value":"timeEntries"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"startTimeSlots"}},{"kind":"Field","name":{"kind":"Name","value":"event"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"time"}},{"kind":"Field","name":{"kind":"Name","value":"site"}},{"kind":"Field","name":{"kind":"Name","value":"description"}}]}},{"kind":"Field","name":{"kind":"Name","value":"plan"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"startTime"}},{"kind":"Field","name":{"kind":"Name","value":"nightConditions"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"iq"}},{"kind":"Field","name":{"kind":"Name","value":"cc"}}]}},{"kind":"Field","name":{"kind":"Name","value":"visits"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"obsId"}},{"kind":"Field","name":{"kind":"Name","value":"endTime"}},{"kind":"Field","name":{"kind":"Name","value":"altitude"}},{"kind":"Field","name":{"kind":"Name","value":"atomEndIdx"}},{"kind":"Field","name":{"kind":"Name","value":"atomStartIdx"}},{"kind":"Field","name":{"kind":"Name","value":"startTime"}},{"kind":"Field","name":{"kind":"Name","value":"instrument"}},{"kind":"Field","name":{"kind":"Name","value":"fpu"}},{"kind":"Field","name":{"kind":"Name","value":"disperser"}},{"kind":"Field","name":{"kind":"Name","value":"filters"}},{"kind":"Field","name":{"kind":"Name","value":"score"}},{"kind":"Field","name":{"kind":"Name","value":"obsClass"}},{"kind":"Field","name":{"kind":"Name","value":"completion"}},{"kind":"Field","name":{"kind":"Name","value":"atomTimes"}},{"kind":"Field","name":{"kind":"Name","value":"peakScore"}},{"kind":"Field","name":{"kind":"Name","value":"requiredConditions"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"iq"}},{"kind":"Field","name":{"kind":"Name","value":"cc"}}]}}]}},{"kind":"Field","name":{"kind":"Name","value":"nightStats"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"timeLoss"}},{"kind":"Field","name":{"kind":"Name","value":"planScore"}},{"kind":"Field","name":{"kind":"Name","value":"nToos"}},{"kind":"Field","name":{"kind":"Name","value":"completionFraction"}},{"kind":"Field","name":{"kind":"Name","value":"programCompletion"}}]}}]}},{"kind":"Field","name":{"kind":"Name","value":"timelossWindows"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"start"}},{"kind":"Field","name":{"kind":"Name","value":"end"}},{"kind":"Field","name":{"kind":"Name","value":"lossType"}}]}}]}}]}}]}}]}},{"kind":"Field","name":{"kind":"Name","value":"plansSummary"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"summary"}},{"kind":"Field","name":{"kind":"Name","value":"metricsPerBand"}}]}}]}},{"kind":"InlineFragment","typeCondition":{"kind":"NamedType","name":{"kind":"Name","value":"NightPlansError"}},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"error"}}]}},{"kind":"InlineFragment","typeCondition":{"kind":"NamedType","name":{"kind":"Name","value":"NewPlansRT"}},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"nightPlans"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"nightIdx"}},{"kind":"Field","name":{"kind":"Name","value":"plansPerSite"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"endTime"}},{"kind":"Field","name":{"kind":"Name","value":"site"}},{"kind":"Field","name":{"kind":"Name","value":"startTime"}},{"kind":"Field","name":{"kind":"Name","value":"visits"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"altitude"}},{"kind":"Field","name":{"kind":"Name","value":"atomEndIdx"}},{"kind":"Field","name":{"kind":"Name","value":"atomStartIdx"}},{"kind":"Field","name":{"kind":"Name","value":"completion"}},{"kind":"Field","name":{"kind":"Name","value":"disperser"}},{"kind":"Field","name":{"kind":"Name","value":"endTime"}},{"kind":"Field","name":{"kind":"Name","value":"filters"}},{"kind":"Field","name":{"kind":"Name","value":"fpu"}},{"kind":"Field","name":{"kind":"Name","value":"instrument"}},{"kind":"Field","name":{"kind":"Name","value":"obsClass"}},{"kind":"Field","name":{"kind":"Name","value":"obsId"}},{"kind":"Field","name":{"kind":"Name","value":"peakScore"}},{"kind":"Field","name":{"kind":"Name","value":"score"}},{"kind":"Field","name":{"kind":"Name","value":"startTime"}},{"kind":"Field","name":{"kind":"Name","value":"requiredConditions"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"cc"}},{"kind":"Field","name":{"kind":"Name","value":"iq"}}]}}]}},{"kind":"Field","name":{"kind":"Name","value":"nightConditions"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"cc"}},{"kind":"Field","name":{"kind":"Name","value":"iq"}}]}},{"kind":"Field","name":{"kind":"Name","value":"nightStats"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"completionFraction"}},{"kind":"Field","name":{"kind":"Name","value":"nToos"}},{"kind":"Field","name":{"kind":"Name","value":"planScore"}},{"kind":"Field","name":{"kind":"Name","value":"programCompletion"}},{"kind":"Field","name":{"kind":"Name","value":"timeLoss"}}]}}]}}]}}]}},{"kind":"InlineFragment","typeCondition":{"kind":"NamedType","name":{"kind":"Name","value":"NightPlansWithEvent"}},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"event"}},{"kind":"Field","name":{"kind":"Name","value":"nightPlans"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"nightIdx"}},{"kind":"Field","name":{"kind":"Name","value":"plansPerSite"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"endTime"}},{"kind":"Field","name":{"kind":"Name","value":"site"}},{"kind":"Field","name":{"kind":"Name","value":"startTime"}},{"kind":"Field","name":{"kind":"Name","value":"visits"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"altitude"}},{"kind":"Field","name":{"kind":"Name","value":"atomEndIdx"}},{"kind":"Field","name":{"kind":"Name","value":"atomStartIdx"}},{"kind":"Field","name":{"kind":"Name","value":"completion"}},{"kind":"Field","name":{"kind":"Name","value":"disperser"}},{"kind":"Field","name":{"kind":"Name","value":"endTime"}},{"kind":"Field","name":{"kind":"Name","value":"filters"}},{"kind":"Field","name":{"kind":"Name","value":"fpu"}},{"kind":"Field","name":{"kind":"Name","value":"instrument"}},{"kind":"Field","name":{"kind":"Name","value":"obsClass"}},{"kind":"Field","name":{"kind":"Name","value":"obsId"}},{"kind":"Field","name":{"kind":"Name","value":"peakScore"}},{"kind":"Field","name":{"kind":"Name","value":"score"}},{"kind":"Field","name":{"kind":"Name","value":"startTime"}},{"kind":"Field","name":{"kind":"Name","value":"requiredConditions"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"cc"}},{"kind":"Field","name":{"kind":"Name","value":"iq"}}]}}]}},{"kind":"Field","name":{"kind":"Name","value":"nightConditions"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"cc"}},{"kind":"Field","name":{"kind":"Name","value":"iq"}}]}},{"kind":"Field","name":{"kind":"Name","value":"nightStats"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"completionFraction"}},{"kind":"Field","name":{"kind":"Name","value":"nToos"}},{"kind":"Field","name":{"kind":"Name","value":"planScore"}},{"kind":"Field","name":{"kind":"Name","value":"programCompletion"}},{"kind":"Field","name":{"kind":"Name","value":"timeLoss"}}]}}]}}]}}]}}]}}]}}]} as unknown as DocumentNode<QueueScheduleSubscription, QueueScheduleSubscriptionVariables>;
export const VersionDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"version"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"version"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"version"}},{"kind":"Field","name":{"kind":"Name","value":"changelog"}}]}}]}}]} as unknown as DocumentNode<VersionQuery, VersionQueryVariables>;
export const ScheduleV2Document = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"scheduleV2"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"scheduleV2"}}]}}]} as unknown as DocumentNode<ScheduleV2Query, ScheduleV2QueryVariables>;
export const OnDemandQueryDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"onDemandQuery"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"onDemandSchedule"}}]}}]} as unknown as DocumentNode<OnDemandQueryQuery, OnDemandQueryQueryVariables>;
export const ScheduleDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"schedule"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"scheduleId"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"startTime"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"endTime"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"sites"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Sites"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"mode"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"SchedulerModes"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"numNightsToSchedule"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"semesterVisibility"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Boolean"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"thesisFactor"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Float"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"power"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"metPower"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Float"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"whaPower"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Float"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"airPower"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Float"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"visPower"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Float"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"programs"}},"type":{"kind":"NonNullType","type":{"kind":"ListType","type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}}}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"schedule"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"scheduleId"},"value":{"kind":"Variable","name":{"kind":"Name","value":"scheduleId"}}},{"kind":"Argument","name":{"kind":"Name","value":"newScheduleInput"},"value":{"kind":"ObjectValue","fields":[{"kind":"ObjectField","name":{"kind":"Name","value":"startTime"},"value":{"kind":"Variable","name":{"kind":"Name","value":"startTime"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"sites"},"value":{"kind":"Variable","name":{"kind":"Name","value":"sites"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"mode"},"value":{"kind":"Variable","name":{"kind":"Name","value":"mode"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"endTime"},"value":{"kind":"Variable","name":{"kind":"Name","value":"endTime"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"thesisFactor"},"value":{"kind":"Variable","name":{"kind":"Name","value":"thesisFactor"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"power"},"value":{"kind":"Variable","name":{"kind":"Name","value":"power"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"metPower"},"value":{"kind":"Variable","name":{"kind":"Name","value":"metPower"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"whaPower"},"value":{"kind":"Variable","name":{"kind":"Name","value":"whaPower"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"airPower"},"value":{"kind":"Variable","name":{"kind":"Name","value":"airPower"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"visPower"},"value":{"kind":"Variable","name":{"kind":"Name","value":"visPower"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"semesterVisibility"},"value":{"kind":"Variable","name":{"kind":"Name","value":"semesterVisibility"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"numNightsToSchedule"},"value":{"kind":"Variable","name":{"kind":"Name","value":"numNightsToSchedule"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"programs"},"value":{"kind":"Variable","name":{"kind":"Name","value":"programs"}}}]}}]}]}}]} as unknown as DocumentNode<ScheduleQuery, ScheduleQueryVariables>;
export const VisibilityAggregatorStatusDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"visibilityAggregatorStatus"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"visibilityAggregatorStatus"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"active"}},{"kind":"Field","name":{"kind":"Name","value":"stale"}},{"kind":"Field","name":{"kind":"Name","value":"holder"}},{"kind":"Field","name":{"kind":"Name","value":"startedAt"}},{"kind":"Field","name":{"kind":"Name","value":"heartbeatAt"}},{"kind":"Field","name":{"kind":"Name","value":"finishedAt"}},{"kind":"Field","name":{"kind":"Name","value":"phase"}},{"kind":"Field","name":{"kind":"Name","value":"progressCurrent"}},{"kind":"Field","name":{"kind":"Name","value":"progressTotal"}},{"kind":"Field","name":{"kind":"Name","value":"progressUnit"}},{"kind":"Field","name":{"kind":"Name","value":"elapsedSeconds"}},{"kind":"Field","name":{"kind":"Name","value":"etaSeconds"}}]}}]}}]} as unknown as DocumentNode<VisibilityAggregatorStatusQuery, VisibilityAggregatorStatusQueryVariables>;
export const VisibilityCoverageDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"visibilityCoverage"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"nightDate"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Date"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"visibilityCoverage"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"nightDate"},"value":{"kind":"Variable","name":{"kind":"Name","value":"nightDate"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"nightDate"}},{"kind":"Field","name":{"kind":"Name","value":"odbReadAt"}},{"kind":"Field","name":{"kind":"Name","value":"expected"}},{"kind":"Field","name":{"kind":"Name","value":"stored"}},{"kind":"Field","name":{"kind":"Name","value":"pending"}},{"kind":"Field","name":{"kind":"Name","value":"missing"}},{"kind":"Field","name":{"kind":"Name","value":"skipped"}},{"kind":"Field","name":{"kind":"Name","value":"isComplete"}},{"kind":"Field","name":{"kind":"Name","value":"pendingKnown"}},{"kind":"Field","name":{"kind":"Name","value":"perProgram"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"key"}},{"kind":"Field","name":{"kind":"Name","value":"expected"}},{"kind":"Field","name":{"kind":"Name","value":"stored"}},{"kind":"Field","name":{"kind":"Name","value":"pending"}},{"kind":"Field","name":{"kind":"Name","value":"missing"}},{"kind":"Field","name":{"kind":"Name","value":"skipped"}}]}},{"kind":"Field","name":{"kind":"Name","value":"perSite"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"key"}},{"kind":"Field","name":{"kind":"Name","value":"expected"}},{"kind":"Field","name":{"kind":"Name","value":"stored"}},{"kind":"Field","name":{"kind":"Name","value":"pending"}},{"kind":"Field","name":{"kind":"Name","value":"missing"}},{"kind":"Field","name":{"kind":"Name","value":"skipped"}}]}}]}}]}}]} as unknown as DocumentNode<VisibilityCoverageQuery, VisibilityCoverageQueryVariables>;
export const ObservationCoverageDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"observationCoverage"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"nightDate"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Date"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"status"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"ObservationStatus"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"site"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"programLabel"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"search"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"limit"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"offset"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"observationCoverage"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"nightDate"},"value":{"kind":"Variable","name":{"kind":"Name","value":"nightDate"}}},{"kind":"Argument","name":{"kind":"Name","value":"status"},"value":{"kind":"Variable","name":{"kind":"Name","value":"status"}}},{"kind":"Argument","name":{"kind":"Name","value":"site"},"value":{"kind":"Variable","name":{"kind":"Name","value":"site"}}},{"kind":"Argument","name":{"kind":"Name","value":"programLabel"},"value":{"kind":"Variable","name":{"kind":"Name","value":"programLabel"}}},{"kind":"Argument","name":{"kind":"Name","value":"search"},"value":{"kind":"Variable","name":{"kind":"Name","value":"search"}}},{"kind":"Argument","name":{"kind":"Name","value":"limit"},"value":{"kind":"Variable","name":{"kind":"Name","value":"limit"}}},{"kind":"Argument","name":{"kind":"Name","value":"offset"},"value":{"kind":"Variable","name":{"kind":"Name","value":"offset"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"total"}},{"kind":"Field","name":{"kind":"Name","value":"nightDate"}},{"kind":"Field","name":{"kind":"Name","value":"odbReadAt"}},{"kind":"Field","name":{"kind":"Name","value":"observations"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"observationId"}},{"kind":"Field","name":{"kind":"Name","value":"programLabel"}},{"kind":"Field","name":{"kind":"Name","value":"site"}},{"kind":"Field","name":{"kind":"Name","value":"targetName"}},{"kind":"Field","name":{"kind":"Name","value":"status"}},{"kind":"Field","name":{"kind":"Name","value":"reason"}}]}}]}}]}}]} as unknown as DocumentNode<ObservationCoverageQuery, ObservationCoverageQueryVariables>;
export const VisibleObservationsDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"visibleObservations"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"site"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"nightDate"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Date"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"limit"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"offset"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"minRemainingMinutes"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"visibleObservations"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"site"},"value":{"kind":"Variable","name":{"kind":"Name","value":"site"}}},{"kind":"Argument","name":{"kind":"Name","value":"nightDate"},"value":{"kind":"Variable","name":{"kind":"Name","value":"nightDate"}}},{"kind":"Argument","name":{"kind":"Name","value":"limit"},"value":{"kind":"Variable","name":{"kind":"Name","value":"limit"}}},{"kind":"Argument","name":{"kind":"Name","value":"offset"},"value":{"kind":"Variable","name":{"kind":"Name","value":"offset"}}},{"kind":"Argument","name":{"kind":"Name","value":"minRemainingMinutes"},"value":{"kind":"Variable","name":{"kind":"Name","value":"minRemainingMinutes"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"site"}},{"kind":"Field","name":{"kind":"Name","value":"nightDate"}},{"kind":"Field","name":{"kind":"Name","value":"total"}},{"kind":"Field","name":{"kind":"Name","value":"totalRemainingMinutes"}},{"kind":"Field","name":{"kind":"Name","value":"observations"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"observationId"}},{"kind":"Field","name":{"kind":"Name","value":"targetName"}},{"kind":"Field","name":{"kind":"Name","value":"remainingMinutes"}},{"kind":"Field","name":{"kind":"Name","value":"remainingMinutesFromNow"}},{"kind":"Field","name":{"kind":"Name","value":"intervals"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"start"}},{"kind":"Field","name":{"kind":"Name","value":"end"}}]}}]}}]}}]}}]} as unknown as DocumentNode<VisibleObservationsQuery, VisibleObservationsQueryVariables>;
export const UpdateWeatherDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"mutation","name":{"kind":"Name","value":"updateWeather"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"weatherInput"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"WeatherInput"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"updateWeather"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"weatherInput"},"value":{"kind":"Variable","name":{"kind":"Name","value":"weatherInput"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"site"}},{"kind":"Field","name":{"kind":"Name","value":"imageQuality"}},{"kind":"Field","name":{"kind":"Name","value":"cloudCover"}},{"kind":"Field","name":{"kind":"Name","value":"windDirection"}},{"kind":"Field","name":{"kind":"Name","value":"windSpeed"}}]}}]}}]} as unknown as DocumentNode<UpdateWeatherMutation, UpdateWeatherMutationVariables>;
export const GetWeatherDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"getWeather"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"weather"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"site"}},{"kind":"Field","name":{"kind":"Name","value":"imageQuality"}},{"kind":"Field","name":{"kind":"Name","value":"cloudCover"}},{"kind":"Field","name":{"kind":"Name","value":"windDirection"}},{"kind":"Field","name":{"kind":"Name","value":"windSpeed"}}]}}]}}]} as unknown as DocumentNode<GetWeatherQuery, GetWeatherQueryVariables>;
export const WeatherUpdatesDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"subscription","name":{"kind":"Name","value":"weatherUpdates"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"weatherUpdates"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"site"}},{"kind":"Field","name":{"kind":"Name","value":"imageQuality"}},{"kind":"Field","name":{"kind":"Name","value":"cloudCover"}},{"kind":"Field","name":{"kind":"Name","value":"windDirection"}},{"kind":"Field","name":{"kind":"Name","value":"windSpeed"}}]}}]}}]} as unknown as DocumentNode<WeatherUpdatesSubscription, WeatherUpdatesSubscriptionVariables>;