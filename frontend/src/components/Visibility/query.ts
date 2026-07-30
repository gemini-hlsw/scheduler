import { graphql } from "@/gql";

/**
 * Every Visibility query runs against the realtime backend: the Sight DB that
 * holds visibility_data is attached to that deployment. Kept as one constant so
 * the second backend can be wired later without touching each call site.
 */
export const REALTIME_CONTEXT = { clientName: "realtimeClient" };

/**
 * Reads Postgres only, so this is the one Visibility query cheap enough to
 * poll while a run is active.
 */
export const aggregatorStatusQuery = graphql(`
  query visibilityAggregatorStatus {
    visibilityAggregatorStatus {
      active
      stale
      holder
      startedAt
      heartbeatAt
      finishedAt
      phase
      progressCurrent
      progressTotal
      progressUnit
      elapsedSeconds
      etaSeconds
    }
  }
`);

/**
 * Reads the ODB live (a paginated sweep), so it takes seconds. Fetch on mount
 * and on explicit refresh only — never poll it.
 */
export const visibilityCoverageQuery = graphql(`
  query visibilityCoverage($nightDate: Date) {
    visibilityCoverage(nightDate: $nightDate) {
      nightDate
      odbReadAt
      expected
      stored
      pending
      missing
      skipped
      isComplete
      pendingKnown
      perProgram {
        key
        expected
        stored
        pending
        missing
        skipped
      }
      perSite {
        key
        expected
        stored
        pending
        missing
        skipped
      }
    }
  }
`);

export const observationCoverageQuery = graphql(`
  query observationCoverage(
    $nightDate: Date
    $status: ObservationStatus
    $site: String
    $programLabel: String
    $search: String
    $limit: Int!
    $offset: Int!
  ) {
    observationCoverage(
      nightDate: $nightDate
      status: $status
      site: $site
      programLabel: $programLabel
      search: $search
      limit: $limit
      offset: $offset
    ) {
      total
      nightDate
      odbReadAt
      observations {
        observationId
        programLabel
        site
        targetName
        status
        skipReason
      }
    }
  }
`);

export const visibleObservationsQuery = graphql(`
  query visibleObservations(
    $site: String!
    $nightDate: Date
    $limit: Int!
    $offset: Int!
    $minRemainingMinutes: Int!
  ) {
    visibleObservations(
      site: $site
      nightDate: $nightDate
      limit: $limit
      offset: $offset
      minRemainingMinutes: $minRemainingMinutes
    ) {
      site
      nightDate
      total
      totalRemainingMinutes
      observations {
        observationId
        targetName
        remainingMinutes
        remainingMinutesFromNow
        intervals {
          start
          end
        }
      }
    }
  }
`);
