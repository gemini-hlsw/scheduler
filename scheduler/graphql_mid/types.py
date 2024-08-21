# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import json
from datetime import datetime, timedelta
from typing import List, FrozenSet, Optional
from zoneinfo import ZoneInfo

import strawberry  # noqa
import astropy.units as u
from astropy.coordinates import Angle
from strawberry.scalars import JSON  # noqa

from lucupy.minimodel import CloudCover, ImageQuality, Site, VariantSnapshot, Conditions

from scheduler.core.eventsqueue.nightchanges import NightlyTimeline
from scheduler.core.plans import Plan, Plans, Visit, NightStats
from scheduler.core.eventsqueue import WeatherChangeEvent
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
        cf = json.dumps(ns.completion_fraction)
        pc = json.dumps(ns.program_completion)
        tl = json.dumps(ns.time_loss)
        return SNightStats(time_loss=tl,
                           plan_score=ns.plan_score,
                           n_toos=ns.n_toos,
                           completion_fraction=cf,
                           program_completion=pc)


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
                      completion=visit.completion)


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
class STimelineEntry:
    start_time_slots: int
    event: str
    plan: SPlan


@strawberry.type
class TimelineEntriesBySite:
    site: Site
    time_entries: List[STimelineEntry]
    eve_twilight: datetime
    morn_twilight: datetime


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
                for entry in timeline.timeline[n_idx][site]:
                    if entry.plan_generated is None:
                        continue
                    e = STimelineEntry(start_time_slots=int(entry.start_time_slot),
                                       event=entry.event.description,
                                       plan=SPlan.from_computed_plan(entry.plan_generated))
                    s_entries.append(e)
                te = TimelineEntriesBySite(site=site,
                                           time_entries=s_entries,
                                           eve_twilight=eve_twi,
                                           morn_twilight=morn_twi)
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

CC = strawberry.enum(CloudCover)
IQ = strawberry.enum(ImageQuality)


@strawberry.type
class NewWeatherChange:
    site: Site
    time: datetime
    description: str
    new_CC: Optional[CC]
    new_IQ: Optional[IQ]
    new_wind_direction: Optional[float]
    new_wind_speed: Optional[float]

    def to_scheduler_event(self) -> WeatherChangeEvent:
        if self.new_wind_direction is None:
            wind_dir = None
        else:
            wind_dir = self.new_wind_direction * u.degrees
        if self.new_wind_speed is None:
            wind_spd = None
        else:
            wind_spd = self.new_wind_speed * (u.m / u.s)
        variant_change = VariantSnapshot(cc=CloudCover[self.new_CC.name] if self.new_CC else None,
                                         iq=ImageQuality[self.new_IQ.name] if self.new_IQ else None,
                                         wind_dir=Angle(wind_dir, unit=u.deg),
                                         wind_spd=wind_spd * (u.m / u.s))
        return WeatherChangeEvent(site=self.site,
                                  time=self.time,
                                  description=self.description,
                                  variant_change=variant_change)


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
