# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import json
from datetime import datetime, timedelta
from typing import List, FrozenSet, Optional

import pytz
import strawberry  # noqa
from strawberry.scalars import JSON

from lucupy.minimodel import (Site, Conditions, ImageQuality,
                              CloudCover, WaterVapor, SkyBackground)

from scheduler.core.eventsqueue.nightchanges import NightTimeline
from scheduler.core.plans import Plan, Plans, Visit, NightStats
from scheduler.core.eventsqueue import WeatherChange
from scheduler.graphql_mid.scalars import SObservationID
from scheduler.config import config


@strawberry.type
class SNightStats:
    """Night stats to display in the UI
    """
    timeloss: str
    plan_score: float
    n_toos: int
    completion_fraction: JSON

    @staticmethod
    def from_computed_night_stats(ns: NightStats) -> 'SNightStats':
        cf = json.dumps(ns.completion_fraction)
        return SNightStats(timeloss=ns.time_loss,
                           plan_score=ns.plan_score,
                           n_toos=ns.n_toos,
                           completion_fraction=cf)


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

    @staticmethod
    def from_computed_visit(visit: Visit, alt_degs: List[float]) -> 'SVisit':
        end_time = visit.start_time + timedelta(minutes=visit.time_slots*config.collector.time_slot_length)
        return SVisit(start_time=visit.start_time.astimezone(pytz.UTC),
                      end_time=end_time.astimezone(pytz.UTC),
                      obs_id=visit.obs_id,
                      atom_start_idx=visit.atom_start_idx,
                      atom_end_idx=visit.atom_end_idx,
                      altitude=alt_degs,
                      instrument=visit.instrument.id if visit.instrument is not None else 'None')


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

    @staticmethod
    def from_computed_plan(plan: Plan) -> 'SPlan':
        return SPlan(
            site=plan.site,
            start_time=plan.start.astimezone(pytz.UTC),
            end_time=plan.end.astimezone(pytz.UTC),
            visits=[SVisit.from_computed_visit(visit, alt) for visit, alt in zip(plan.visits, plan.alt_degs)],
            night_stats=SNightStats.from_computed_night_stats(plan.night_stats)
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
            plans_per_site=[plans for plans in self.plans_per_site if plans.site == site])


@strawberry.type
class STimelineEntry:
    start_time_slots: int
    event: str
    plan: SPlan


@strawberry.type
class TimelineEntriesBySite:
    site: Site
    time_entries: List[STimelineEntry]


@strawberry.type
class SNightInTimeline:
    night_index: int
    time_entries_by_site: List[TimelineEntriesBySite]


@strawberry.type
class SNightTimelines:
    night_timeline: List[SNightInTimeline]

    @staticmethod
    def from_computed_timelines(timeline: NightTimeline) -> 'SNightTimelines':
        timelines = []
        for n_idx in timeline.timeline:
            s_timeline_entries = []
            for site in timeline.timeline[n_idx]:
                s_entries = []
                for entry in timeline.timeline[n_idx][site]:

                    e = STimelineEntry(start_time_slots=int(entry.start_time_slots),
                                       event=entry.event.reason,
                                       plan=SPlan.from_computed_plan(entry.plan_generated))
                    s_entries.append(e)
                te = TimelineEntriesBySite(site=site, time_entries=s_entries)
                s_timeline_entries.append(te)
            sn = SNightInTimeline(night_index=n_idx, time_entries_by_site=s_timeline_entries)
            timelines.append(sn)
        return SNightTimelines(night_timeline=timelines)


@strawberry.type
class NewNightPlans:
    night_plans: SNightTimelines
    plans_summary: JSON


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

SB = strawberry.enum(SkyBackground)
CC = strawberry.enum(CloudCover)
WV = strawberry.enum(WaterVapor)
IQ = strawberry.enum(ImageQuality)


@strawberry.type
class NewWeatherChange:
    start: datetime
    reason: str
    new_CC: Optional[CC]
    new_SB: Optional[SB]
    new_WV: Optional[WV]
    new_IQ: Optional[IQ]

    def to_scheduler_event(self) -> WeatherChange:
        c = Conditions(cc=CloudCover[self.new_CC.name] if self.new_CC else None,
                       sb=SkyBackground[self.new_SB.name] if self.new_SB else None,
                       wv=WaterVapor[self.new_WV.name] if self.new_WV else None,
                       iq=ImageQuality[self.new_IQ.name] if self.new_IQ else None)
        return WeatherChange(start=self.start,
                             reason=self.reason,
                             new_conditions=c
                             )


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


EventsAddedResponse = EventsAddedSuccess | EventsAddedError
