# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

"""
Time accounting of interrupted (partial) observations.

An event -- a weather closure or a fault -- can cut a visit short. The atoms that did not
finish must not be charged, and the observation must not be marked OBSERVED: the Selector
skips OBSERVED observations, so marking one prematurely loses its remaining atoms forever.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Sequence, Tuple

import astropy.units as u
import pytest
from astropy.time import TimeDelta
from lucupy.minimodel import (AndOption, Atom, GROUP_NONE_ID, Group, GroupID, NightIndex, Observation,
                              ObservationClass, ObservationID, ObservationMode, ObservationStatus, Priority, Program,
                              ProgramID, ProgramMode, ProgramTypes, QAState, ROOT_GROUP_ID, SetupTimeType, Site)
from lucupy.types import ZeroTime

from scheduler.core.components.collector import Collector
from scheduler.core.plans import Plan, Plans, Visit
from scheduler.time_accountant import TimeAccountant

_SITE = Site.GN
_NIGHT_IDX = NightIndex(0)
_PROGRAM_ID = ProgramID('GN-2018B-Q-101')
_START = datetime(2018, 10, 1, 20, 0, 0)


# The Collector ClassVars are global state shared with the (module-scoped) collector fixtures
# in conftest, so they are swapped out through monkeypatch rather than assigned directly.
@pytest.fixture(autouse=True)
def _isolated_collector_tables(monkeypatch):
    monkeypatch.setattr(Collector, '_observations', {}, raising=False)
    monkeypatch.setattr(Collector, '_programs', {}, raising=False)


def _collector(slot_minutes: float = 1.0) -> Collector:
    """A Collector holding only what time_accounting reads.

    __post_init__ builds night events and loads resource files; time_accounting touches
    none of that, so it is bypassed.
    """
    collector = Collector.__new__(Collector)
    collector.time_slot_length = TimeDelta(slot_minutes * u.min)
    collector.time_accountant = TimeAccountant(frozenset({_SITE}), [_NIGHT_IDX])
    return collector


def _atom(atom_id: int, exec_seconds: int, obs_class: ObservationClass, observed: bool = False) -> Atom:
    exec_time = timedelta(seconds=exec_seconds)
    is_partner = obs_class == ObservationClass.PARTNERCAL
    return Atom(id=atom_id,
                exec_time=exec_time,
                prog_time=ZeroTime if is_partner else exec_time,
                part_time=exec_time if is_partner else ZeroTime,
                program_used=ZeroTime,
                partner_used=ZeroTime,
                not_charged=ZeroTime,
                observed=observed,
                qa_state=QAState.PASS if observed else QAState.NONE,
                guide_state=True,
                resources=frozenset(),
                wavelengths=frozenset(),
                obs_mode=ObservationMode.IMAGING)


def _observation(obs_id: str,
                 acq_seconds: int,
                 atom_specs: Sequence[Tuple[int, bool]],
                 obs_class: ObservationClass = ObservationClass.SCIENCE,
                 status: ObservationStatus = ObservationStatus.READY) -> Observation:
    """atom_specs is a sequence of (exec_seconds, already_observed) pairs."""
    return Observation(id=ObservationID(obs_id),
                       internal_id=obs_id,
                       order=0,
                       title=obs_id,
                       site=_SITE,
                       status=status,
                       active=True,
                       priority=Priority.MEDIUM,
                       setuptime_type=SetupTimeType.FULL,
                       acq_overhead=timedelta(seconds=acq_seconds),
                       obs_class=obs_class,
                       targets=[],
                       guiding={},
                       sequence=[_atom(idx, secs, obs_class, observed)
                                 for idx, (secs, observed) in enumerate(atom_specs)],
                       belongs_to=_PROGRAM_ID,
                       constraints=None)


def _group(group_id: GroupID, children, number_to_observe: int) -> Group:
    return Group(id=group_id,
                 program_id=_PROGRAM_ID,
                 group_name=group_id.id,
                 parent_id=ROOT_GROUP_ID,
                 previous_id=GROUP_NONE_ID,
                 next_id=GROUP_NONE_ID,
                 number_to_observe=number_to_observe,
                 number_observed=0,
                 delay_min=ZeroTime,
                 delay_max=ZeroTime,
                 active=True,
                 children=children,
                 group_option=AndOption.CONSEC_ORDERED)


def _observation_group(obs: Observation) -> Group:
    """The trivial AND group wrapping a single observation, as the program providers build it."""
    return _group(GroupID(obs.id.id), obs, 1)


def _register(observations: List[Observation], scheduling_group: bool = False) -> List[Group]:
    """Populate the Collector tables and return the top-level groups, in root order."""
    obs_groups = [_observation_group(obs) for obs in observations]
    if scheduling_group:
        top_level = [_group(GroupID('sched'), obs_groups, len(obs_groups))]
    else:
        top_level = obs_groups

    root = _group(ROOT_GROUP_ID, top_level, len(top_level))
    Collector._programs[_PROGRAM_ID] = Program(id=_PROGRAM_ID,
                                               internal_id=_PROGRAM_ID.id,
                                               semester=None,
                                               thesis=False,
                                               mode=ProgramMode.QUEUE,
                                               type=ProgramTypes.Q,
                                               start=_START,
                                               end=_START + timedelta(days=180),
                                               allocated_time=frozenset(),
                                               used_time=frozenset(),
                                               root_group=root)
    for obs in observations:
        Collector._observations[obs.id] = (obs, None)
    return top_level


def _visit(obs: Observation,
           atom_start: int,
           atom_end: int,
           start_time_slot: int,
           time_slots: int) -> Visit:
    return Visit(start_time=_START,
                 observation=obs,
                 atom_start_idx=atom_start,
                 atom_end_idx=atom_end,
                 start_time_slot=start_time_slot,
                 time_slots=time_slots,
                 score=1.0,
                 peak_score=1.0,
                 step_start_idx=None,
                 step_count=None,
                 completion=f'{atom_end + 1}/{len(obs.sequence)}',
                 atom_times=[])


def _plans(visits: List[Visit], slot_minutes: float = 1.0) -> Plans:
    plans = Plans(night_events={}, night_conditions={}, night_idx=_NIGHT_IDX)
    plan = Plan(start=_START,
                end=_START + timedelta(hours=10),
                time_slot_length=timedelta(minutes=slot_minutes),
                site=_SITE,
                _time_slots_left=600,
                conditions=None)
    plan.visits = list(visits)
    plans[_SITE] = plan
    return plans


def _account(collector: Collector, plans: Plans, bound: Optional[int] = None) -> None:
    end_timeslot_bounds = None if bound is None else {_SITE: bound}
    collector.time_accounting(plans=plans,
                              sites=frozenset({_SITE}),
                              end_timeslot_bounds=end_timeslot_bounds)


def test_truncated_visit_leaves_observation_ongoing():
    """An event at slot 12 cuts a 3-atom visit after its second atom.

    acq=2min + atoms of 5min each, so the atoms end at slots 6, 11 and 16. The last atom
    never ran, so the observation must stay schedulable.
    """
    obs = _observation('GN-2018B-Q-101-1', acq_seconds=120, atom_specs=[(300, False)] * 3)
    group = _register([obs])[0]
    collector = _collector()

    _account(collector, _plans([_visit(obs, 0, 2, start_time_slot=0, time_slots=17)]), bound=12)

    assert [atom.observed for atom in obs.sequence] == [True, True, False]
    assert obs.status is ObservationStatus.ONGOING
    assert group.number_observed == 0


def test_full_night_marks_observation_observed():
    """Whole-night accounting is unchanged: everything ran, so the observation completes."""
    obs = _observation('GN-2018B-Q-101-1', acq_seconds=120, atom_specs=[(300, False)] * 3)
    group = _register([obs])[0]
    collector = _collector()

    _account(collector, _plans([_visit(obs, 0, 2, start_time_slot=0, time_slots=17)]))

    assert all(atom.observed for atom in obs.sequence)
    assert obs.status is ObservationStatus.OBSERVED
    assert group.number_observed == 1
    # The acquisition overhead is charged to the first atom.
    assert obs.sequence[0].program_used == timedelta(seconds=300 + 120)


def test_truncated_partner_cal_is_inactivated_not_observed():
    """A partly-executed standard is still swept to INACTIVE, not reported as OBSERVED.

    The 'we only needed one standard' semantics of the INACTIVE sweep are deliberately
    kept; what must not happen is the observation being recorded as fully observed.
    """
    obs = _observation('GN-2018B-Q-101-2', acq_seconds=120, atom_specs=[(300, False)] * 3,
                       obs_class=ObservationClass.PARTNERCAL)
    group = _register([obs])[0]
    collector = _collector()

    _account(collector, _plans([_visit(obs, 0, 2, start_time_slot=0, time_slots=17)]), bound=12)

    assert obs.status is ObservationStatus.INACTIVE
    assert obs.sequence[2].observed is False
    assert group.number_observed == 0


def test_charge_group_with_no_charged_atoms_leaves_status_untouched():
    """Nothing ran tonight, so the status is not time accounting's to change.

    The observation resumes at atom 1 after a short (1 min) atom 0 observed on an earlier
    night. charge_group is True only because collector.py:702 measures sequence[0] rather
    than the first *unobserved* atom (a separate, still-open TODO), but the bound at slot 5
    precedes atom 1's end at slot 11, so nothing is charged.
    """
    obs = _observation('GN-2018B-Q-101-1',
                       acq_seconds=120,
                       atom_specs=[(60, True), (600, False), (600, False)],
                       status=ObservationStatus.ONGOING)
    group = _register([obs])[0]
    collector = _collector()

    _account(collector, _plans([_visit(obs, 1, 2, start_time_slot=0, time_slots=22)]), bound=5)

    assert [atom.observed for atom in obs.sequence] == [True, False, False]
    assert obs.status is ObservationStatus.ONGOING
    assert group.number_observed == 0


def test_fractional_acquisition_charges_last_atom_on_full_night():
    """A 90s acquisition must not cost the visit its final atom.

    Converting the acquisition and the sequence separately takes two ceilings,
    ceil(90s) + ceil(150s) = 2 + 3 = 5 slots, where GreedyMax sized the visit as a single
    ceil(90s + 150s) = 4 slots. Under the two-ceiling form the last atom ends outside its
    own visit and falls past the charge window on a full, uninterrupted night.
    """
    obs = _observation('GN-2018B-Q-101-1', acq_seconds=90, atom_specs=[(75, False)] * 2)
    group = _register([obs])[0]
    collector = _collector()

    _account(collector, _plans([_visit(obs, 0, 1, start_time_slot=0, time_slots=4)]))

    assert all(atom.observed for atom in obs.sequence)
    assert obs.status is ObservationStatus.OBSERVED
    assert group.number_observed == 1


def test_interrupted_scheduling_group_is_not_charged():
    """A scheduling group cut short is charged to not_charged, and no status is touched."""
    obs_a = _observation('GN-2018B-Q-101-1', acq_seconds=120, atom_specs=[(300, False)] * 2)
    obs_b = _observation('GN-2018B-Q-101-2', acq_seconds=120, atom_specs=[(300, False)] * 2)
    _register([obs_a, obs_b], scheduling_group=True)
    collector = _collector()

    plans = _plans([_visit(obs_a, 0, 1, start_time_slot=0, time_slots=12),
                    _visit(obs_b, 0, 1, start_time_slot=12, time_slots=12)])
    _account(collector, plans, bound=18)

    for obs in (obs_a, obs_b):
        assert obs.status is ObservationStatus.READY
        assert all(atom.program_used == ZeroTime for atom in obs.sequence)
        assert all(not atom.observed for atom in obs.sequence)
    # The group ran into the bound, so the time it consumed is not charged to the program.
    assert obs_a.not_charged() > ZeroTime


def test_two_visits_of_one_observation_in_a_night_charge_both():
    """An observation continued later the same night is charged for both of its visits.

    The two visits land in separate GroupVisits -- consecutive visits are merged only for
    scheduling groups -- so they are accounted in turn: the first leaves the observation
    ONGOING, the second completes it. Each visit pays its own acquisition overhead.
    """
    obs = _observation('GN-2018B-Q-101-1', acq_seconds=120, atom_specs=[(300, False)] * 3)
    group = _register([obs])[0]
    collector = _collector()

    _account(collector, _plans([_visit(obs, 0, 1, start_time_slot=0, time_slots=12),
                                _visit(obs, 2, 2, start_time_slot=12, time_slots=7)]))

    assert all(atom.observed for atom in obs.sequence)
    # Acquisition is charged to the first atom of each visit, so atoms 0 and 2 carry it.
    assert [atom.program_used for atom in obs.sequence] == [timedelta(seconds=420),
                                                            timedelta(seconds=300),
                                                            timedelta(seconds=420)]
    assert obs.status is ObservationStatus.OBSERVED
    # Two visits, but one observation completed.
    assert group.number_observed == 1


def test_second_pass_completes_interrupted_observation():
    """The atoms left behind by an interruption are charged and complete the observation."""
    obs = _observation('GN-2018B-Q-101-1', acq_seconds=120, atom_specs=[(300, False)] * 3)
    group = _register([obs])[0]
    collector = _collector()

    _account(collector, _plans([_visit(obs, 0, 2, start_time_slot=0, time_slots=17)]), bound=12)
    assert obs.status is ObservationStatus.ONGOING
    assert group.number_observed == 0

    # Replan: the observation resumes at its first unobserved atom and runs to the end.
    _account(collector, _plans([_visit(obs, 2, 2, start_time_slot=0, time_slots=7)]))

    assert all(atom.observed for atom in obs.sequence)
    assert obs.status is ObservationStatus.OBSERVED
    assert group.number_observed == 1
