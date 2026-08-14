# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from datetime import datetime, timedelta, date
from enum import Enum
from typing import List, FrozenSet, Optional
from zoneinfo import ZoneInfo

import strawberry  # noqa
from typing import Annotated, Union
from strawberry.scalars import JSON  # noqa

from lucupy.minimodel import CloudCover, ImageQuality, Site, VariantSnapshot, Conditions

from scheduler.core.events.queue import NightlyTimeline
from scheduler.core.plans import Plan, Plans, Visit, NightStats
from scheduler.core.statscalculator import RunSummary
from scheduler.graphql_mid.scalars import SObservationID
from scheduler.config import config


@strawberry.type
class SNightStats:
    """Night stats to display in the UI
    """
    time_loss: JSON
    plan_score: float
    n_toos: int
    completion_fraction: JSON
    program_completion: JSON

    @staticmethod
    def from_computed_night_stats(ns: NightStats) -> 'SNightStats':
        return SNightStats(time_loss=ns.time_loss,
                           plan_score=ns.plan_score,
                           n_toos=ns.n_toos,
                           completion_fraction=ns.completion_fraction,
                           program_completion=ns.program_completion)


@strawberry.type
class SRunSummary:
    summary: JSON
    metrics_per_band: JSON

    @staticmethod
    def from_computed_run_summary(summary: RunSummary) -> 'SRunSummary':
        return SRunSummary(summary=summary.summary, metrics_per_band=summary.metrics_per_band)


@strawberry.type
class SConditions:
    iq: str
    cc: str

    @staticmethod
    def from_computed_conditions(variant: VariantSnapshot | Conditions):
        return SConditions(iq=variant.iq.name, cc=variant.cc.name)


@strawberry.type
class SVisit:
    """
    Represents a visit as part of a nightly Plan at a Site.
    """
    start_time: datetime
    end_time: datetime
    obs_id: SObservationID
    atom_start_idx: int
    atom_end_idx: int
    altitude: List[float]
    instrument: str
    fpu: str
    disperser: str
    filters: List[str]
    required_conditions: SConditions
    obs_class: str
    score: float
    peak_score: float
    completion: str
    atom_times: List[int]

    @staticmethod
    def from_computed_visit(visit: Visit, alt_degs: List[float]) -> 'SVisit':
        utc = ZoneInfo('UTC')
        end_time = visit.start_time + timedelta(minutes=visit.time_slots * config.collector.time_slot_length)
        return SVisit(start_time=visit.start_time.astimezone(utc),
                      end_time=end_time.astimezone(utc),
                      obs_id=visit.obs_id,
                      atom_start_idx=visit.atom_start_idx,
                      atom_end_idx=visit.atom_end_idx,
                      altitude=alt_degs,
                      instrument=visit.instrument.id if visit.instrument is not None else 'None',
                      fpu=visit.fpu.id if visit.fpu is not None else 'None',
                      disperser=visit.disperser.id if visit.disperser is not None else 'None',
                      filters=[f.id for f in visit.filters] if visit.filters is not None else [],
                      required_conditions=SConditions.from_computed_conditions(visit.obs_conditions),
                      score=visit.score,
                      peak_score=visit.peak_score,
                      obs_class=visit.obs_class.name,
                      completion=visit.completion,
                      atom_times=visit.atom_times)


@strawberry.type
class SPlan:
    """
    A nightly Plan for a specific site.
    """
    site: strawberry.enum(Site)
    start_time: datetime
    end_time: datetime
    visits: List[SVisit]
    night_stats: SNightStats
    night_conditions: SConditions

    @staticmethod
    def from_computed_plan(plan: Plan) -> 'SPlan':
        utc = ZoneInfo('UTC')
        return SPlan(
            site=plan.site,
            start_time=plan.start.astimezone(utc),
            end_time=plan.end.astimezone(utc),
            visits=[SVisit.from_computed_visit(visit, alt) for visit, alt in zip(plan.visits, plan.alt_degs)],
            night_stats=SNightStats.from_computed_night_stats(plan.night_stats),
            night_conditions=SConditions.from_computed_conditions(plan.conditions)
        )


@strawberry.type
class SPlans:
    """
    For a given night, a collection of Plan for each Site.
    """
    # TODO: Change this to date in UTC
    night_idx: int
    plans_per_site: List[SPlan]

    @staticmethod
    def from_computed_plans(plans: Plans, sites: FrozenSet[Site]) -> 'SPlans':
        return SPlans(
            night_idx=plans.night_idx,
            plans_per_site=[SPlan.from_computed_plan(plans[site]) for site in sites])

    def for_site(self, site: Site) -> 'SPlans':
        return SPlans(
            night_idx=self.night_idx,
            plans_per_site=[plans for plans in self.plans_per_site if plans is not None and plans.site == site])

@strawberry.type
class Event:
    site: Site
    time: datetime
    description: str


@strawberry.type
class STimeLossWindow:
    start: datetime
    end: Optional[datetime]
    loss_type: str

@strawberry.type
class STimelineEntry:
    start_time_slots: int
    event: Event
    plan: SPlan
    timeloss_windows: List[STimeLossWindow]


@strawberry.type
class TimelineEntriesBySite:
    site: Site
    time_entries: List[STimelineEntry]
    eve_twilight: datetime
    morn_twilight: datetime
    time_losses: JSON


@strawberry.type
class SNightInTimeline:
    night_index: int
    time_entries_by_site: List[TimelineEntriesBySite]


@strawberry.type
class SNightTimelines:
    night_timeline: List[SNightInTimeline]

    @staticmethod
    def from_computed_timelines(timeline: NightlyTimeline) -> 'SNightTimelines':
        timelines = []
        for n_idx in timeline.timeline:
            s_timeline_entries = []
            for site in timeline.timeline[n_idx]:
                s_entries = []
                eve_twi = timeline.timeline[n_idx][site][0].event.time
                morn_twi = timeline.timeline[n_idx][site][-1].event.time
                time_losses = timeline.time_losses[n_idx][site]
                for entry in timeline.timeline[n_idx][site]:
                    if entry.plan_generated is None:
                        continue
                    e = STimelineEntry(start_time_slots=int(entry.start_time_slot),
                                       event=Event(site=entry.event.site,
                                                   time=entry.event.time,
                                                   description=entry.event.description),
                                       plan=SPlan.from_computed_plan(entry.plan_generated),
                                       timeloss_windows=entry.timeloss_windows)
                    s_entries.append(e)
                te = TimelineEntriesBySite(site=site,
                                           time_entries=s_entries,
                                           eve_twilight=eve_twi,
                                           morn_twilight=morn_twi,
                                           time_losses=time_losses)
                s_timeline_entries.append(te)
            sn = SNightInTimeline(night_index=n_idx, time_entries_by_site=s_timeline_entries)
            timelines.append(sn)
        return SNightTimelines(night_timeline=timelines)

    @staticmethod
    def from_computed_stitched_timelines(timeline: NightlyTimeline) -> 'SNightTimelines':
        timelines = []
        for n_idx in timeline.stitched_timeline:
            s_timeline_entries = []
            for site in timeline.stitched_timeline[n_idx]:
                s_entries = []
                # eve_twi = timeline.stitched_timeline[n_idx][site][0].event.time
                # morn_twi = timeline.stitched_timeline[n_idx][site][-1].event.time
                time_losses = timeline.time_losses[n_idx][site]
                for entry in timeline.stitched_timeline[n_idx][site]:
                    if entry.plan_generated is None:
                        continue
                    e = STimelineEntry(start_time_slots=int(entry.start_time_slot),
                                        event=Event(site=entry.event.site if entry.event.site is not None else site,
                                                    time=entry.event.time if entry.event.time is not None else timeline.night_length[n_idx][site].start,
                                                    description=entry.event.description),
                                        plan=SPlan.from_computed_plan(entry.plan_generated),
                                        timeloss_windows=entry.timeloss_windows)
                    s_entries.append(e)
                te = TimelineEntriesBySite(site=site,
                                            time_entries=s_entries,
                                            eve_twilight=timeline.night_length[n_idx][site].start,
                                            morn_twilight=timeline.night_length[n_idx][site].end,
                                            time_losses=time_losses)
                s_timeline_entries.append(te)
            sn = SNightInTimeline(night_index=n_idx, time_entries_by_site=s_timeline_entries)
            timelines.append(sn)
        return SNightTimelines(night_timeline=timelines)

@strawberry.type
class NewNightPlans:
    night_plans: SNightTimelines
    plans_summary: SRunSummary

@strawberry.type
class NewPlansRT:
    night_plans: SPlans

@strawberry.type
class NightPlansError:
    error: str

@strawberry.type
class NightPlansWithEvent:
    night_plans: SPlans
    event: str

NightPlansResponse = Annotated[Union[NewNightPlans, NightPlansError], strawberry.union("NightPlansResponse")]

NightPlansResponseRT = Annotated[Union[NightPlansResponse, NewPlansRT, NightPlansWithEvent], strawberry.union("NightPlansResponseRT")]

@strawberry.type
class NewScheduleSuccess:
    """
    Success response for creating a new schedule.
    """
    success: bool


@strawberry.type
class NewScheduleError:
    """
    Error response for creating a new schedule.
    """
    error: str


@strawberry.type
class ChangeOriginSuccess:
    """
    Success response for creating a new schedule.
    """
    from_origin: str
    to_origin: str


@strawberry.type
class SourceFileHandlerResponse:
    """
    Error response for missing implementation from
    files in a service.
    """
    service: str
    loaded: bool
    msg: str


NewScheduleResponse = NewScheduleSuccess | NewScheduleError

CC = strawberry.enum(CloudCover)
IQ = strawberry.enum(ImageQuality)


@strawberry.type
class RankerParameters:
    """
    Ranker Parameters used for modifying the scoring algorithm in the Ranker
    component. These are only used on Multi Night modes: VALIDATION and SIMULATION.
    """
    thesis_factor: Optional[float] = 1.1
    power: Optional[int] = 2
    met_power: Optional[float] = 1.0
    vis_power: Optional[float] = 1.0
    wha_power: Optional[float] = 1.0
    air_power: Optional[float] = 0.0


@strawberry.type
class EventsAddedSuccess:
    """
    Success response for creating a new schedule.
    """
    success: bool
    added_event: str


@strawberry.type
class EventsAddedError:
    """
    Error response for creating a new schedule.
    """
    error: str

@strawberry.type
class Version:
    version: str
    changelog: List[str]

EventsAddedResponse = EventsAddedSuccess | EventsAddedError

from scheduler.engine.params import BuildParameters, NightTimes




@strawberry.experimental.pydantic.input(model=NightTimes, all_fields=True)
class NightTimesInput:
    pass

@strawberry.input
class SiteNightTimesEntry:
    site: strawberry.enum(Site)
    night_times: NightTimesInput

@strawberry.experimental.pydantic.input(model=BuildParameters)
class BuildParametersInput:
    night_times: Optional[List[SiteNightTimesEntry]] = None
    visibility_start: strawberry.auto
    visibility_end: strawberry.auto
    program_list: strawberry.auto

    def to_pydantic(self) -> BuildParameters:
        """Convert to Pydantic model"""
        night_times_dict = None
        if self.night_times:
            night_times_dict = {
                Site(entry.site): NightTimes(
                    night_start=entry.night_times.night_start,
                    night_end=entry.night_times.night_end
                )
                for entry in self.night_times
            }

        return BuildParameters(
            night_times=night_times_dict,
            visibility_start=self.visibility_start,
            visibility_end=self.visibility_end,
            program_list=self.program_list
        )

@strawberry.type
class NightTimesResponse:
    site: str
    start: Optional[datetime]
    end: Optional[datetime]

@strawberry.type
class BuildParametersResponse:
    night_times: Optional[List[NightTimesResponse]]
    visibility_start: Optional[datetime]
    visibility_end: Optional[datetime]
    program_list: Optional[List[str]]


@strawberry.type
class AvailableProgram:
    id: str
    ref_label: str


@strawberry.type
class VisibilityAggregatorStatus:
    """Current state of the background visibility-aggregator cron."""
    active: bool
    stale: bool
    holder: Optional[str]
    started_at: Optional[str]
    heartbeat_at: Optional[str]
    finished_at: Optional[str]
    detail: Optional[str]  # JSON-encoded progress detail
    # Parsed out of `detail`. All null outside the phases that report progress.
    phase: Optional[str] = None
    progress_current: Optional[int] = None
    progress_total: Optional[int] = None
    progress_unit: Optional[str] = None  # "targets" / "nights"
    elapsed_seconds: Optional[float] = None
    eta_seconds: Optional[float] = None

    @staticmethod
    def from_service(status) -> 'VisibilityAggregatorStatus':
        return VisibilityAggregatorStatus(
            active=status.active,
            stale=status.stale,
            holder=status.holder,
            started_at=status.started_at,
            heartbeat_at=status.heartbeat_at,
            finished_at=status.finished_at,
            detail=status.detail,
            phase=status.phase,
            progress_current=status.progress_current,
            progress_total=status.progress_total,
            progress_unit=status.progress_unit,
            elapsed_seconds=status.elapsed_seconds,
            eta_seconds=status.eta_seconds,
        )


# --- Visibility coverage ----------------------------------------------------
#
# Only reference labels (`G-…`) appear on the wire. The ODB GIDs (`o-…`, `p-…`)
# are matching keys internal to the visibility_status services: an operator
# cannot look one up, so no field here exposes one.

@strawberry.enum
class ObservationStatus(Enum):
    """Whether an observation's visibility is stored, stale, absent, or N/A."""
    STORED = "STORED"
    PENDING = "PENDING"    # ODB inputs changed; awaiting recomputation
    MISSING = "MISSING"
    SKIPPED = "SKIPPED"    # can never be stored (e.g. non-sidereal)


@strawberry.type
class GroupCoverage:
    """Coverage counts for one group.

    ``key`` is the program label under `perProgram` and the site key under
    `perSite`.
    """
    key: str
    expected: int
    stored: int
    pending: int
    missing: int
    skipped: int

    @staticmethod
    def from_service(group) -> 'GroupCoverage':
        return GroupCoverage(
            key=group.key,
            expected=group.expected,
            stored=group.stored,
            pending=group.pending,
            missing=group.missing,
            skipped=group.skipped,
        )


@strawberry.type
class VisibilityCoverage:
    """Whether the Sight DB holds visibility for everything the ODB expects."""
    night_date: Optional[date]
    odb_read_at: Optional[datetime]
    expected: int
    stored: int
    pending: int
    missing: int
    skipped: int
    is_complete: bool
    # False when the ODB change probe failed, so `pending` is not authoritative.
    pending_known: bool
    per_program: List[GroupCoverage]
    per_site: List[GroupCoverage]

    @staticmethod
    def from_service(summary) -> 'VisibilityCoverage':
        return VisibilityCoverage(
            night_date=summary.night_date,
            odb_read_at=summary.odb_read_at,
            expected=summary.expected,
            stored=summary.stored,
            pending=summary.pending,
            missing=summary.missing,
            skipped=summary.skipped,
            is_complete=summary.is_complete,
            pending_known=summary.pending_known,
            per_program=[GroupCoverage.from_service(g) for g in summary.per_program],
            per_site=[GroupCoverage.from_service(g) for g in summary.per_site],
        )


@strawberry.type
class ObservationCoverage:
    """One observation's coverage on the night being examined."""
    observation_id: str      # reference label, e.g. G-2026A-0001-Q-0001
    program_label: str
    site: Optional[str]
    target_name: Optional[str]
    status: ObservationStatus
    # Why the status is what it is, as a token the UI renders in words (see
    # services/visibility_status/reasons.py). Null only for STORED.
    reason: Optional[str]

    @staticmethod
    def from_service(row) -> 'ObservationCoverage':
        return ObservationCoverage(
            observation_id=row.observation_id,
            program_label=row.program_label,
            site=row.site,
            target_name=row.target_name,
            status=ObservationStatus(row.status),
            reason=row.reason,
        )


@strawberry.type
class ObservationCoveragePage:
    observations: List[ObservationCoverage]
    total: int
    night_date: Optional[date]
    odb_read_at: Optional[datetime]

    @staticmethod
    def from_service(page) -> 'ObservationCoveragePage':
        return ObservationCoveragePage(
            observations=[
                ObservationCoverage.from_service(o) for o in page.observations
            ],
            total=page.total,
            night_date=page.night_date,
            odb_read_at=page.odb_read_at,
        )


@strawberry.type
class VisibleInterval:
    """A contiguous window of visibility, in UTC."""
    start: datetime
    end: datetime


@strawberry.type
class VisibleObservation:
    """What is observable for one observation on one night."""
    observation_id: str
    site: str
    target_name: Optional[str]
    night_date: date
    remaining_minutes: int             # over the whole night
    remaining_minutes_from_now: int    # only what is still ahead
    intervals: List[VisibleInterval]

    @staticmethod
    def from_service(observation) -> 'VisibleObservation':
        return VisibleObservation(
            observation_id=observation.observation_id,
            site=observation.site,
            target_name=observation.target_name,
            night_date=observation.night_date,
            remaining_minutes=observation.remaining_minutes,
            remaining_minutes_from_now=observation.remaining_minutes_from_now,
            intervals=[
                VisibleInterval(start=i.start, end=i.end)
                for i in observation.intervals
            ],
        )


@strawberry.type
class VisibleObservationsPage:
    site: str
    night_date: date
    observations: List[VisibleObservation]
    total: int
    # Summed over this page only, not the whole night.
    total_remaining_minutes: int

    @staticmethod
    def from_service(page) -> 'VisibleObservationsPage':
        return VisibleObservationsPage(
            site=page.site,
            night_date=page.night_date,
            observations=[
                VisibleObservation.from_service(o) for o in page.observations
            ],
            total=page.total,
            total_remaining_minutes=page.total_remaining_minutes,
        )
