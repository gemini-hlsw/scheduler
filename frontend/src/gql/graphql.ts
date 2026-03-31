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

export type CreateNewScheduleRtInput = {
  airPower?: InputMaybe<Scalars['Float']>;
  cloudCover?: InputMaybe<Scalars['Float']>;
  endTime: Scalars['String'];
  imageQuality?: InputMaybe<Scalars['Float']>;
  metPower?: InputMaybe<Scalars['Float']>;
  nightEndTime: Scalars['String'];
  nightStartTime: Scalars['String'];
  power?: InputMaybe<Scalars['Int']>;
  programs?: InputMaybe<Array<Scalars['String']>>;
  sites: Scalars['Sites'];
  startTime: Scalars['String'];
  thesisFactor?: InputMaybe<Scalars['Float']>;
  visPower?: InputMaybe<Scalars['Float']>;
  whaPower?: InputMaybe<Scalars['Float']>;
  windDirection?: InputMaybe<Scalars['Float']>;
  windSpeed?: InputMaybe<Scalars['Float']>;
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

export type Query = {
  __typename?: 'Query';
  availablePrograms: Array<AvailableProgram>;
  schedule: Scalars['String'];
  scheduleV2: Scalars['String'];
  version: Version;
  weather?: Maybe<Array<Maybe<Weather>>>;
};


export type QueryScheduleArgs = {
  newScheduleInput: CreateNewScheduleInput;
  scheduleId: Scalars['String'];
};


export type QueryScheduleV2Args = {
  newScheduleRtInput: CreateNewScheduleRtInput;
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

export type STimelineEntry = {
  __typename?: 'STimelineEntry';
  event: Scalars['String'];
  plan: SPlan;
  startTimeSlots: Scalars['Int'];
};

export type SVisit = {
  __typename?: 'SVisit';
  altitude: Array<Scalars['Float']>;
  atomEndIdx: Scalars['Int'];
  atomStartIdx: Scalars['Int'];
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

export type QueueScheduleSubscriptionVariables = Exact<{
  scheduleId: Scalars['String'];
}>;


export type QueueScheduleSubscription = { __typename?: 'Subscription', queueSchedule: { __typename: 'NewNightPlans', nightPlans: { __typename?: 'SNightTimelines', nightTimeline: Array<{ __typename?: 'SNightInTimeline', nightIndex: number, timeEntriesBySite: Array<{ __typename?: 'TimelineEntriesBySite', site: Site, mornTwilight: any, eveTwilight: any, timeLosses: any, timeEntries: Array<{ __typename?: 'STimelineEntry', startTimeSlots: number, event: string, plan: { __typename?: 'SPlan', startTime: any, nightConditions: { __typename?: 'SConditions', iq: string, cc: string }, visits: Array<{ __typename?: 'SVisit', obsId: any, endTime: any, altitude: Array<number>, atomEndIdx: number, atomStartIdx: number, startTime: any, instrument: string, fpu: string, disperser: string, filters: Array<string>, score: number, obsClass: string, completion: string, peakScore: number, requiredConditions: { __typename?: 'SConditions', iq: string, cc: string } }>, nightStats: { __typename?: 'SNightStats', timeLoss: any, planScore: number, nToos: number, completionFraction: any, programCompletion: any } } }> }> }> }, plansSummary: { __typename?: 'SRunSummary', summary: any, metricsPerBand: any } } | { __typename: 'NewPlansRT', nightPlans: { __typename?: 'SPlans', nightIdx: number, plansPerSite: Array<{ __typename?: 'SPlan', endTime: any, site: Site, startTime: any, visits: Array<{ __typename?: 'SVisit', altitude: Array<number>, atomEndIdx: number, atomStartIdx: number, completion: string, disperser: string, endTime: any, filters: Array<string>, fpu: string, instrument: string, obsClass: string, obsId: any, peakScore: number, score: number, startTime: any, requiredConditions: { __typename?: 'SConditions', cc: string, iq: string } }>, nightConditions: { __typename?: 'SConditions', cc: string, iq: string }, nightStats: { __typename?: 'SNightStats', completionFraction: any, nToos: number, planScore: number, programCompletion: any, timeLoss: any } }> } } | { __typename: 'NightPlansError', error: string } | { __typename: 'NightPlansWithEvent', event: string, nightPlans: { __typename?: 'SPlans', nightIdx: number, plansPerSite: Array<{ __typename?: 'SPlan', endTime: any, site: Site, startTime: any, visits: Array<{ __typename?: 'SVisit', altitude: Array<number>, atomEndIdx: number, atomStartIdx: number, completion: string, disperser: string, endTime: any, filters: Array<string>, fpu: string, instrument: string, obsClass: string, obsId: any, peakScore: number, score: number, startTime: any, requiredConditions: { __typename?: 'SConditions', cc: string, iq: string } }>, nightConditions: { __typename?: 'SConditions', cc: string, iq: string }, nightStats: { __typename?: 'SNightStats', completionFraction: any, nToos: number, planScore: number, programCompletion: any, timeLoss: any } }> } } };

export type VersionQueryVariables = Exact<{ [key: string]: never; }>;


export type VersionQuery = { __typename?: 'Query', version: { __typename?: 'Version', version: string, changelog: Array<string> } };

export type ScheduleV2QueryVariables = Exact<{
  startTime: Scalars['String'];
  endTime: Scalars['String'];
  nightStartTime: Scalars['String'];
  nightEndTime: Scalars['String'];
  sites: Scalars['Sites'];
  imageQuality: Scalars['Float'];
  cloudCover: Scalars['Float'];
  windSpeed: Scalars['Float'];
  windDirection: Scalars['Float'];
  thesisFactor?: InputMaybe<Scalars['Float']>;
  power?: InputMaybe<Scalars['Int']>;
  metPower?: InputMaybe<Scalars['Float']>;
  whaPower?: InputMaybe<Scalars['Float']>;
  airPower?: InputMaybe<Scalars['Float']>;
  visPower?: InputMaybe<Scalars['Float']>;
  programs: Array<Scalars['String']> | Scalars['String'];
}>;


export type ScheduleV2Query = { __typename?: 'Query', scheduleV2: string };

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
export const QueueScheduleDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"subscription","name":{"kind":"Name","value":"queueSchedule"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"scheduleId"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"queueSchedule"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"scheduleId"},"value":{"kind":"Variable","name":{"kind":"Name","value":"scheduleId"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"__typename"}},{"kind":"InlineFragment","typeCondition":{"kind":"NamedType","name":{"kind":"Name","value":"NewNightPlans"}},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"nightPlans"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"nightTimeline"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"nightIndex"}},{"kind":"Field","name":{"kind":"Name","value":"timeEntriesBySite"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"site"}},{"kind":"Field","name":{"kind":"Name","value":"mornTwilight"}},{"kind":"Field","name":{"kind":"Name","value":"eveTwilight"}},{"kind":"Field","name":{"kind":"Name","value":"timeLosses"}},{"kind":"Field","name":{"kind":"Name","value":"timeEntries"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"startTimeSlots"}},{"kind":"Field","name":{"kind":"Name","value":"event"}},{"kind":"Field","name":{"kind":"Name","value":"plan"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"startTime"}},{"kind":"Field","name":{"kind":"Name","value":"nightConditions"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"iq"}},{"kind":"Field","name":{"kind":"Name","value":"cc"}}]}},{"kind":"Field","name":{"kind":"Name","value":"visits"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"obsId"}},{"kind":"Field","name":{"kind":"Name","value":"endTime"}},{"kind":"Field","name":{"kind":"Name","value":"altitude"}},{"kind":"Field","name":{"kind":"Name","value":"atomEndIdx"}},{"kind":"Field","name":{"kind":"Name","value":"atomStartIdx"}},{"kind":"Field","name":{"kind":"Name","value":"startTime"}},{"kind":"Field","name":{"kind":"Name","value":"instrument"}},{"kind":"Field","name":{"kind":"Name","value":"fpu"}},{"kind":"Field","name":{"kind":"Name","value":"disperser"}},{"kind":"Field","name":{"kind":"Name","value":"filters"}},{"kind":"Field","name":{"kind":"Name","value":"score"}},{"kind":"Field","name":{"kind":"Name","value":"obsClass"}},{"kind":"Field","name":{"kind":"Name","value":"completion"}},{"kind":"Field","name":{"kind":"Name","value":"peakScore"}},{"kind":"Field","name":{"kind":"Name","value":"requiredConditions"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"iq"}},{"kind":"Field","name":{"kind":"Name","value":"cc"}}]}}]}},{"kind":"Field","name":{"kind":"Name","value":"nightStats"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"timeLoss"}},{"kind":"Field","name":{"kind":"Name","value":"planScore"}},{"kind":"Field","name":{"kind":"Name","value":"nToos"}},{"kind":"Field","name":{"kind":"Name","value":"completionFraction"}},{"kind":"Field","name":{"kind":"Name","value":"programCompletion"}}]}}]}}]}}]}}]}}]}},{"kind":"Field","name":{"kind":"Name","value":"plansSummary"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"summary"}},{"kind":"Field","name":{"kind":"Name","value":"metricsPerBand"}}]}}]}},{"kind":"InlineFragment","typeCondition":{"kind":"NamedType","name":{"kind":"Name","value":"NightPlansError"}},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"error"}}]}},{"kind":"InlineFragment","typeCondition":{"kind":"NamedType","name":{"kind":"Name","value":"NewPlansRT"}},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"nightPlans"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"nightIdx"}},{"kind":"Field","name":{"kind":"Name","value":"plansPerSite"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"endTime"}},{"kind":"Field","name":{"kind":"Name","value":"site"}},{"kind":"Field","name":{"kind":"Name","value":"startTime"}},{"kind":"Field","name":{"kind":"Name","value":"visits"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"altitude"}},{"kind":"Field","name":{"kind":"Name","value":"atomEndIdx"}},{"kind":"Field","name":{"kind":"Name","value":"atomStartIdx"}},{"kind":"Field","name":{"kind":"Name","value":"completion"}},{"kind":"Field","name":{"kind":"Name","value":"disperser"}},{"kind":"Field","name":{"kind":"Name","value":"endTime"}},{"kind":"Field","name":{"kind":"Name","value":"filters"}},{"kind":"Field","name":{"kind":"Name","value":"fpu"}},{"kind":"Field","name":{"kind":"Name","value":"instrument"}},{"kind":"Field","name":{"kind":"Name","value":"obsClass"}},{"kind":"Field","name":{"kind":"Name","value":"obsId"}},{"kind":"Field","name":{"kind":"Name","value":"peakScore"}},{"kind":"Field","name":{"kind":"Name","value":"score"}},{"kind":"Field","name":{"kind":"Name","value":"startTime"}},{"kind":"Field","name":{"kind":"Name","value":"requiredConditions"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"cc"}},{"kind":"Field","name":{"kind":"Name","value":"iq"}}]}}]}},{"kind":"Field","name":{"kind":"Name","value":"nightConditions"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"cc"}},{"kind":"Field","name":{"kind":"Name","value":"iq"}}]}},{"kind":"Field","name":{"kind":"Name","value":"nightStats"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"completionFraction"}},{"kind":"Field","name":{"kind":"Name","value":"nToos"}},{"kind":"Field","name":{"kind":"Name","value":"planScore"}},{"kind":"Field","name":{"kind":"Name","value":"programCompletion"}},{"kind":"Field","name":{"kind":"Name","value":"timeLoss"}}]}}]}}]}}]}},{"kind":"InlineFragment","typeCondition":{"kind":"NamedType","name":{"kind":"Name","value":"NightPlansWithEvent"}},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"event"}},{"kind":"Field","name":{"kind":"Name","value":"nightPlans"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"nightIdx"}},{"kind":"Field","name":{"kind":"Name","value":"plansPerSite"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"endTime"}},{"kind":"Field","name":{"kind":"Name","value":"site"}},{"kind":"Field","name":{"kind":"Name","value":"startTime"}},{"kind":"Field","name":{"kind":"Name","value":"visits"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"altitude"}},{"kind":"Field","name":{"kind":"Name","value":"atomEndIdx"}},{"kind":"Field","name":{"kind":"Name","value":"atomStartIdx"}},{"kind":"Field","name":{"kind":"Name","value":"completion"}},{"kind":"Field","name":{"kind":"Name","value":"disperser"}},{"kind":"Field","name":{"kind":"Name","value":"endTime"}},{"kind":"Field","name":{"kind":"Name","value":"filters"}},{"kind":"Field","name":{"kind":"Name","value":"fpu"}},{"kind":"Field","name":{"kind":"Name","value":"instrument"}},{"kind":"Field","name":{"kind":"Name","value":"obsClass"}},{"kind":"Field","name":{"kind":"Name","value":"obsId"}},{"kind":"Field","name":{"kind":"Name","value":"peakScore"}},{"kind":"Field","name":{"kind":"Name","value":"score"}},{"kind":"Field","name":{"kind":"Name","value":"startTime"}},{"kind":"Field","name":{"kind":"Name","value":"requiredConditions"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"cc"}},{"kind":"Field","name":{"kind":"Name","value":"iq"}}]}}]}},{"kind":"Field","name":{"kind":"Name","value":"nightConditions"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"cc"}},{"kind":"Field","name":{"kind":"Name","value":"iq"}}]}},{"kind":"Field","name":{"kind":"Name","value":"nightStats"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"completionFraction"}},{"kind":"Field","name":{"kind":"Name","value":"nToos"}},{"kind":"Field","name":{"kind":"Name","value":"planScore"}},{"kind":"Field","name":{"kind":"Name","value":"programCompletion"}},{"kind":"Field","name":{"kind":"Name","value":"timeLoss"}}]}}]}}]}}]}}]}}]}}]} as unknown as DocumentNode<QueueScheduleSubscription, QueueScheduleSubscriptionVariables>;
export const VersionDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"version"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"version"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"version"}},{"kind":"Field","name":{"kind":"Name","value":"changelog"}}]}}]}}]} as unknown as DocumentNode<VersionQuery, VersionQueryVariables>;
export const ScheduleV2Document = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"scheduleV2"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"startTime"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"endTime"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"nightStartTime"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"nightEndTime"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"sites"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Sites"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"imageQuality"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Float"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"cloudCover"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Float"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"windSpeed"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Float"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"windDirection"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Float"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"thesisFactor"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Float"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"power"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"metPower"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Float"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"whaPower"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Float"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"airPower"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Float"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"visPower"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Float"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"programs"}},"type":{"kind":"NonNullType","type":{"kind":"ListType","type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}}}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"scheduleV2"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"newScheduleRtInput"},"value":{"kind":"ObjectValue","fields":[{"kind":"ObjectField","name":{"kind":"Name","value":"startTime"},"value":{"kind":"Variable","name":{"kind":"Name","value":"startTime"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"endTime"},"value":{"kind":"Variable","name":{"kind":"Name","value":"endTime"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"nightStartTime"},"value":{"kind":"Variable","name":{"kind":"Name","value":"nightStartTime"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"nightEndTime"},"value":{"kind":"Variable","name":{"kind":"Name","value":"nightEndTime"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"sites"},"value":{"kind":"Variable","name":{"kind":"Name","value":"sites"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"imageQuality"},"value":{"kind":"Variable","name":{"kind":"Name","value":"imageQuality"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"cloudCover"},"value":{"kind":"Variable","name":{"kind":"Name","value":"cloudCover"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"windSpeed"},"value":{"kind":"Variable","name":{"kind":"Name","value":"windSpeed"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"windDirection"},"value":{"kind":"Variable","name":{"kind":"Name","value":"windDirection"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"thesisFactor"},"value":{"kind":"Variable","name":{"kind":"Name","value":"thesisFactor"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"power"},"value":{"kind":"Variable","name":{"kind":"Name","value":"power"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"metPower"},"value":{"kind":"Variable","name":{"kind":"Name","value":"metPower"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"whaPower"},"value":{"kind":"Variable","name":{"kind":"Name","value":"whaPower"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"airPower"},"value":{"kind":"Variable","name":{"kind":"Name","value":"airPower"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"visPower"},"value":{"kind":"Variable","name":{"kind":"Name","value":"visPower"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"programs"},"value":{"kind":"Variable","name":{"kind":"Name","value":"programs"}}}]}}]}]}}]} as unknown as DocumentNode<ScheduleV2Query, ScheduleV2QueryVariables>;
export const ScheduleDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"schedule"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"scheduleId"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"startTime"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"endTime"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"sites"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Sites"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"mode"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"SchedulerModes"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"numNightsToSchedule"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"semesterVisibility"}},"type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"Boolean"}}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"thesisFactor"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Float"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"power"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Int"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"metPower"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Float"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"whaPower"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Float"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"airPower"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Float"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"visPower"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"Float"}}},{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"programs"}},"type":{"kind":"NonNullType","type":{"kind":"ListType","type":{"kind":"NonNullType","type":{"kind":"NamedType","name":{"kind":"Name","value":"String"}}}}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"schedule"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"scheduleId"},"value":{"kind":"Variable","name":{"kind":"Name","value":"scheduleId"}}},{"kind":"Argument","name":{"kind":"Name","value":"newScheduleInput"},"value":{"kind":"ObjectValue","fields":[{"kind":"ObjectField","name":{"kind":"Name","value":"startTime"},"value":{"kind":"Variable","name":{"kind":"Name","value":"startTime"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"sites"},"value":{"kind":"Variable","name":{"kind":"Name","value":"sites"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"mode"},"value":{"kind":"Variable","name":{"kind":"Name","value":"mode"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"endTime"},"value":{"kind":"Variable","name":{"kind":"Name","value":"endTime"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"thesisFactor"},"value":{"kind":"Variable","name":{"kind":"Name","value":"thesisFactor"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"power"},"value":{"kind":"Variable","name":{"kind":"Name","value":"power"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"metPower"},"value":{"kind":"Variable","name":{"kind":"Name","value":"metPower"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"whaPower"},"value":{"kind":"Variable","name":{"kind":"Name","value":"whaPower"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"airPower"},"value":{"kind":"Variable","name":{"kind":"Name","value":"airPower"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"visPower"},"value":{"kind":"Variable","name":{"kind":"Name","value":"visPower"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"semesterVisibility"},"value":{"kind":"Variable","name":{"kind":"Name","value":"semesterVisibility"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"numNightsToSchedule"},"value":{"kind":"Variable","name":{"kind":"Name","value":"numNightsToSchedule"}}},{"kind":"ObjectField","name":{"kind":"Name","value":"programs"},"value":{"kind":"Variable","name":{"kind":"Name","value":"programs"}}}]}}]}]}}]} as unknown as DocumentNode<ScheduleQuery, ScheduleQueryVariables>;
export const UpdateWeatherDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"mutation","name":{"kind":"Name","value":"updateWeather"},"variableDefinitions":[{"kind":"VariableDefinition","variable":{"kind":"Variable","name":{"kind":"Name","value":"weatherInput"}},"type":{"kind":"NamedType","name":{"kind":"Name","value":"WeatherInput"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"updateWeather"},"arguments":[{"kind":"Argument","name":{"kind":"Name","value":"weatherInput"},"value":{"kind":"Variable","name":{"kind":"Name","value":"weatherInput"}}}],"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"site"}},{"kind":"Field","name":{"kind":"Name","value":"imageQuality"}},{"kind":"Field","name":{"kind":"Name","value":"cloudCover"}},{"kind":"Field","name":{"kind":"Name","value":"windDirection"}},{"kind":"Field","name":{"kind":"Name","value":"windSpeed"}}]}}]}}]} as unknown as DocumentNode<UpdateWeatherMutation, UpdateWeatherMutationVariables>;
export const GetWeatherDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"query","name":{"kind":"Name","value":"getWeather"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"weather"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"site"}},{"kind":"Field","name":{"kind":"Name","value":"imageQuality"}},{"kind":"Field","name":{"kind":"Name","value":"cloudCover"}},{"kind":"Field","name":{"kind":"Name","value":"windDirection"}},{"kind":"Field","name":{"kind":"Name","value":"windSpeed"}}]}}]}}]} as unknown as DocumentNode<GetWeatherQuery, GetWeatherQueryVariables>;
export const WeatherUpdatesDocument = {"kind":"Document","definitions":[{"kind":"OperationDefinition","operation":"subscription","name":{"kind":"Name","value":"weatherUpdates"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"weatherUpdates"},"selectionSet":{"kind":"SelectionSet","selections":[{"kind":"Field","name":{"kind":"Name","value":"site"}},{"kind":"Field","name":{"kind":"Name","value":"imageQuality"}},{"kind":"Field","name":{"kind":"Name","value":"cloudCover"}},{"kind":"Field","name":{"kind":"Name","value":"windDirection"}},{"kind":"Field","name":{"kind":"Name","value":"windSpeed"}}]}}]}}]} as unknown as DocumentNode<WeatherUpdatesSubscription, WeatherUpdatesSubscriptionVariables>;