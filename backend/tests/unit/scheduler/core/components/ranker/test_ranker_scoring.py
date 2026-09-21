# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""Characterization tests for the Rankers, written to pin GSCHED-1031.

`AdditiveRanker` was a verbatim copy of `DefaultRanker` that changed exactly one
expression, and GSCHED-1031 collapses the copy into the shared `Ranker` base. There were no
ranker tests at all, so these were written and made green *before* that refactor: the
`_GOLDEN_*` constants below are the pre-refactor output, byte for byte. That is the whole
proof that hoisting the shared code did not move a single score.

Regenerate the goldens only if a scoring change is *intended*, and say so in the commit:

    REGENERATE_RANKER_GOLDENS=1 pytest <this file> -k regenerate

The two rankers read met_power / vis_power / wha_power differently: exponents under
DefaultRanker, linear weights under AdditiveRanker. Both readings are pinned here.
"""

import os
from dataclasses import fields

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import Angle
from lucupy.minimodel import ALL_SITES, Band, ObservationStatus, Priority
from lucupy.types import MinMax
from numpy.testing import assert_allclose

from scheduler.core.components.ranker import (AdditiveRanker, DefaultRanker, Ranker,
                                              RankerBandParameters, RankerParameters)
from scheduler.core.components.ranker import additive as additive_mod
from scheduler.core.components.ranker import default as default_mod

from .conftest import (NIGHTS, N_SLOTS, SITE, FakeGroup, FakeObservation, FakeProgram,
                       group_data_map, night_configurations)

# Pre-refactor scores for the baseline scenario in `conftest.collector` with default
# RankerParameters. Hex of `scores[night].tobytes()`, so the comparison is bit-exact with no
# float formatting in between. Both nights are identical by construction.
_GOLDEN_DEFAULT = (
    '0000000000000000e2c4dd9ee362f73feea5af65f50d1f4025e055d249d42540'
    '25e055d249d42540f2a5af65f50d1f40f1c4dd9ee362f73f0000000000000000'
)
_GOLDEN_ADDITIVE = (
    '0ad7a3703dca2240bc15d846c5902340b09bdf42c0e82640ab5ee3c0bd942840'
    'ab5ee3c0bd942840b19bdf42c0e82640bc15d846c59023400ad7a3703dca2240'
)


def _ranker(cls, collector, params=None, nights=NIGHTS):
    return cls(collector, nights, frozenset({SITE}), params=params or RankerParameters())


def _score(cls, collector, params=None, obs=None, program=None, nc=None, nights=NIGHTS):
    obs = obs or FakeObservation()
    program = program or FakeProgram()
    return _ranker(cls, collector, params, nights).score_observation(
        program, obs, nc or night_configurations(), nights)


# --------------------------------------------------------------------------------------
# Bit-exact snapshots: the actual regression guard.
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize('cls, golden', [(DefaultRanker, _GOLDEN_DEFAULT),
                                         (AdditiveRanker, _GOLDEN_ADDITIVE)])
def test_scores_are_byte_identical_to_pre_refactor(cls, golden, collector):
    scores = _score(cls, collector)

    for night_idx in NIGHTS:
        assert scores[night_idx].tobytes().hex() == golden, (
            f'{cls.__name__} night {night_idx} scores moved; see this module docstring')


# --------------------------------------------------------------------------------------
# The one expression that differs between the two rankers.
# --------------------------------------------------------------------------------------

def _expected_terms(collector, params, night_idx):
    """Rebuild the metric / visibility / wha terms independently of the Ranker."""
    ti = collector.target_info[night_idx]
    program, obs = FakeProgram(), FakeObservation()
    cplt = (program.total_used(obs.band) + obs.exec_time() - obs.total_used()) / \
        program.total_awarded(obs.band)
    metric = DefaultRanker(collector, NIGHTS, frozenset({SITE}), params=params).metric_slope(
        np.array([cplt]), np.array([obs.band.value]), np.array([0.8]), program.thesis)[0][0]
    # GS latitude and a -30 deg target are 0.24 deg apart, so the <40 deg coefficients apply.
    c = params.dec_diff_less_40
    wha = c[0] + c[1] * ti.hourangle / u.hourangle + \
        (c[2] / u.hourangle ** 2) * ti.hourangle ** 2
    wha = np.asarray(np.where(wha <= 0., 0., wha))
    return metric, ti.rem_visibility_frac, wha, np.min(ti.airmass)


@pytest.mark.parametrize('params', [
    RankerParameters(),
    RankerParameters(met_power=2.0, vis_power=0.5, wha_power=1.5),
    RankerParameters(met_power=1.0, vis_power=1.0, wha_power=1.0, air_power=2.0),
])
def test_default_ranker_multiplies_a_product_of_powers(params, collector):
    scores = _score(DefaultRanker, collector, params)

    for night_idx in NIGHTS:
        metric, vis, wha, air = _expected_terms(collector, params, night_idx)
        expected = ((metric ** params.met_power) * (vis ** params.vis_power) *
                    (wha ** params.wha_power) / (air ** params.air_power))
        assert_allclose(scores[night_idx], expected, rtol=1e-12)


@pytest.mark.parametrize('params', [
    RankerParameters(),
    RankerParameters(met_power=2.0, vis_power=0.5, wha_power=1.5),
    RankerParameters(met_power=1.0, vis_power=1.0, wha_power=1.0, air_power=2.0),
])
def test_additive_ranker_sums_weighted_terms(params, collector):
    scores = _score(AdditiveRanker, collector, params)

    for night_idx in NIGHTS:
        metric, vis, wha, air = _expected_terms(collector, params, night_idx)
        expected = ((metric * params.met_power) + (vis * params.vis_power) +
                    (wha * params.wha_power)) / (air ** params.air_power)
        assert_allclose(scores[night_idx], expected, rtol=1e-12)


def test_a_zero_wha_slot_zeroes_default_but_not_additive(collector):
    """The deliberate consequence of the additive formula, worth pinning explicitly.

    Multiplying, a slot outside the hour-angle window kills the score. Summing, the metric
    and visibility terms still contribute, so the observation stays schedulable there.
    """
    zeroed = np.flatnonzero(_expected_terms(collector, RankerParameters(), NIGHTS[0])[2] == 0.)
    assert zeroed.size, 'the baseline scenario must contain at least one clamped slot'

    assert np.all(_score(DefaultRanker, collector)[NIGHTS[0]][zeroed] == 0.)
    assert np.all(_score(AdditiveRanker, collector)[NIGHTS[0]][zeroed] > 0.)


# --------------------------------------------------------------------------------------
# Shared plumbing: identical for both rankers, so parametrize over both.
# --------------------------------------------------------------------------------------

BOTH = pytest.mark.parametrize('cls', [DefaultRanker, AdditiveRanker])


@BOTH
def test_wha_is_clamped_at_zero(cls, collector):
    assert np.all(_score(cls, collector)[NIGHTS[0]] >= 0.)


@BOTH
def test_slots_outside_the_altitude_limits_score_zero(cls, collector, make_target_info):
    # Default limits are [18, 88] deg; put the first slot below and the last above.
    alt = np.full(N_SLOTS, 55.0)
    alt[0], alt[-1] = 10.0, 89.0
    collector.target_info = {n: make_target_info(alt_deg=alt) for n in NIGHTS}

    scores = _score(cls, collector)[NIGHTS[0]]
    assert scores[0] == 0. and scores[-1] == 0.
    assert np.all(scores[1:-1] > 0.), 'only the out-of-limit slots may be zeroed'


@BOTH
def test_a_night_without_target_info_stays_zero(cls, collector, make_target_info):
    """Sight only populates target_info for visible (obs, night) pairs.

    Iterating night_indices blindly would KeyError on the rest; the pre-padded score array
    must come back untouched for those nights instead.
    """
    collector.target_info = {NIGHTS[0]: make_target_info()}

    scores = _score(cls, collector)
    assert np.all(scores[NIGHTS[1]] == 0.)
    assert np.any(scores[NIGHTS[0]] > 0.)


@BOTH
def test_no_target_info_at_all_returns_all_zero_scores(cls, collector):
    collector.get_target_info = lambda _obs_id: None

    scores = _score(cls, collector)
    assert sorted(scores) == sorted(NIGHTS)
    assert all(np.all(scores[n] == 0.) for n in NIGHTS)


@BOTH
def test_only_visible_slots_are_scored(cls, collector, make_target_info):
    visible = np.array([2, 3, 4])
    collector.target_info = {n: make_target_info(visibility_slot_idx=visible) for n in NIGHTS}

    scores = _score(cls, collector)[NIGHTS[0]]
    assert np.all(scores[visible] > 0.)
    assert np.all(np.delete(scores, visible) == 0.)


@BOTH
@pytest.mark.parametrize('kwargs, factor', [
    ({'preimaging': True}, RankerParameters().preimaging_factor),
    ({'status': ObservationStatus.ONGOING}, RankerParameters().ongoing_factor),
    # priority scales as 1 + (priority - mean) / priority_factor; mean_priority() is 1.0.
    ({'priority': Priority.HIGH}, 1. + (Priority.HIGH.value - 1.) / RankerParameters().priority_factor),
    ({'priority': Priority.LOW}, 1. + (Priority.LOW.value - 1.) / RankerParameters().priority_factor),
])
def test_scale_factors_multiply_the_whole_night(cls, kwargs, factor, collector):
    baseline = _score(cls, collector)[NIGHTS[0]]

    scaled = _score(cls, collector, obs=FakeObservation(**kwargs))[NIGHTS[0]]
    assert_allclose(scaled, baseline * factor, rtol=1e-12)


@BOTH
def test_program_priority_boosts_the_score(cls, collector):
    baseline = _score(cls, collector)[NIGHTS[0]]

    boosted = _score(cls, collector, nc=night_configurations(NIGHTS))[NIGHTS[0]]
    assert_allclose(boosted, baseline * RankerParameters().program_priority, rtol=1e-12)


@BOTH
def test_the_last_nights_program_priority_is_applied_to_every_night(cls, collector):
    """KNOWN BUG, pinned deliberately (see vis_drift.md 11 and 14.8).

    `scale_factor *= prog_priority[night_idx]` reads the night_idx left over from the
    altitude loop, i.e. always the last night. So marking only the LAST night as a priority
    program boosts both nights, and marking only the FIRST boosts neither. Harmless today
    because every production caller scores one night per call. GSCHED-1031 preserves this
    as-is; fixing it changes scores and needs its own ticket. When that lands, this test is
    the one to invert.
    """
    baseline = {n: _score(cls, collector)[n] for n in NIGHTS}
    boost = RankerParameters().program_priority

    last_only = _score(cls, collector, nc=night_configurations((NIGHTS[-1],)))
    for night_idx in NIGHTS:
        assert_allclose(last_only[night_idx], baseline[night_idx] * boost, rtol=1e-12)

    first_only = _score(cls, collector, nc=night_configurations((NIGHTS[0],)))
    for night_idx in NIGHTS:
        assert_allclose(first_only[night_idx], baseline[night_idx], rtol=1e-12)


@BOTH
def test_a_far_southern_target_uses_the_wide_dec_coefficients(cls, collector, make_target_info):
    """dec_diff >= 40 deg switches the hour-angle coefficient set."""
    params = RankerParameters(dec_diff_less_40=np.array([3., 0., -0.08]),
                              dec_diff=np.array([1., 0., -0.01]))
    collector.target_info = {n: make_target_info(dec_deg=30.0) for n in NIGHTS}

    ti = collector.target_info[NIGHTS[0]]
    wha = params.dec_diff[0] + (params.dec_diff[2] / u.hourangle ** 2) * ti.hourangle ** 2
    assert np.all(np.asarray(wha) > 0.), 'the wide coefficients must not clamp here'

    scores = _score(cls, collector, params)[NIGHTS[0]]
    assert np.all(scores > 0.)


# --------------------------------------------------------------------------------------
# metric_slope: shared, and the one method a consumer calls directly
# (statscalculator.py:117).
# --------------------------------------------------------------------------------------

@BOTH
@pytest.mark.parametrize('band', [Band.BAND1, Band.BAND2, Band.BAND3, Band.BAND4])
@pytest.mark.parametrize('power', [1, 2])
@pytest.mark.parametrize('thesis', [False, True])
@pytest.mark.parametrize('completion', [0.0, 1e-9, 0.5, 0.9, 1.0, 1.4])
def test_metric_slope_matches_its_piecewise_definition(cls, band, power, thesis, completion,
                                                       collector):
    params = RankerParameters(power=power)
    ranker = _ranker(cls, collector, params)
    bp = ranker.band_params[band]

    metric, slope = ranker.metric_slope(np.array([completion]), np.array([band.value]),
                                        np.array([0.8]), thesis)

    # Band 3 takes xb from the b3min argument instead of the band parameters.
    xb = 0.8 if band == Band.BAND3 else bp.xb
    b2 = (xb * (bp.m1 - bp.m2) + bp.xb0 + bp.b1) if power == 1 else (bp.b2 + bp.xb0 + bp.b1)
    if completion <= 1.e-7:
        expected, expected_slope = 0.0, 0.0
    elif completion < xb:
        expected = bp.m1 * completion ** power + bp.b1
        expected_slope = power * bp.m1 * completion ** (power - 1.0)
    elif completion < 1.0:
        expected, expected_slope = bp.m2 * completion + b2, bp.m2
    else:
        expected, expected_slope = bp.m2 + b2 + bp.xc0, bp.m2

    assert_allclose(metric, [expected + (params.thesis_factor if thesis else 0.)], rtol=1e-12)
    assert_allclose(slope, [expected_slope], rtol=1e-12)


@BOTH
def test_metric_slope_rejects_mismatched_lengths(cls, collector):
    with pytest.raises(ValueError, match='Incompatible lengths'):
        _ranker(cls, collector).metric_slope(np.array([0.5, 0.6]), np.array([Band.BAND1.value]),
                                             np.array([0.8]), False)


@BOTH
def test_explicit_band_params_are_used(cls, collector):
    """GSCHED-1031 fix: passing band_params used to leave the attribute unset.

    `if band_params is None: self.band_params = ...` had no else branch, so an explicit map
    was silently dropped and metric_slope then raised AttributeError. No caller passes it
    today, which is why fixing it cannot change any current output.
    """
    flat = RankerBandParameters(m1=0.0, b1=7.0, m2=0.0, b2=0.0, xb=0.8, xb0=0.0, xc0=0.0)
    custom = {band: flat for band in [Band.BAND1, Band.BAND2, Band.BAND3, Band.BAND4]}

    ranker = cls(collector, NIGHTS, frozenset({SITE}), params=RankerParameters(),
                 band_params=custom)

    assert ranker.band_params is custom
    metric, _ = ranker.metric_slope(np.array([0.5]), np.array([Band.BAND2.value]),
                                    np.array([0.8]), False)
    assert_allclose(metric, [7.0])


# --------------------------------------------------------------------------------------
# Group scoring: shared, dispatched from the base class.
# --------------------------------------------------------------------------------------

@BOTH
def test_and_group_takes_the_max_unless_a_child_is_zero(cls, collector):
    """score_combiner: max(children), but a single 0 zeroes the slot.

    Child arrays must be N_SLOTS long: the group accumulator is pre-shaped (0, N_SLOTS)
    from the collector's night events, and np.append is strict about the row width.
    """
    a = [1., 0., 3., 4., 2., 9., 0., 5.]
    b = [2., 5., 0., 4., 7., 1., 0., 5.]

    scores = _ranker(cls, collector).score_group(
        FakeGroup(), group_data_map({'a': {NIGHTS[0]: a}, 'b': {NIGHTS[0]: b}}))

    assert_allclose(scores[NIGHTS[0]], [2., 0., 0., 4., 7., 9., 0., 5.])


@BOTH
def test_and_group_rejects_multiple_sites(cls, collector):
    with pytest.raises(ValueError, match='has too many sites'):
        _ranker(cls, collector).score_group(
            FakeGroup(group_sites=ALL_SITES),
            group_data_map({'a': {NIGHTS[0]: [1.]}, 'b': {NIGHTS[0]: [1.]}}))


@BOTH
def test_or_groups_are_not_implemented(cls, collector):
    with pytest.raises(NotImplementedError):
        _ranker(cls, collector).score_group(FakeGroup(and_group=False), group_data_map({}))


@BOTH
def test_score_group_refuses_a_non_group(cls, collector):
    class NotAGroup(FakeGroup):
        def is_and_group(self): return False
        def is_or_group(self): return False

    with pytest.raises(ValueError, match='can only score groups'):
        _ranker(cls, collector).score_group(NotAGroup(), group_data_map({}))


# --------------------------------------------------------------------------------------
# Structure: what stops the duplication growing back. These fail before GSCHED-1031.
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize('name', ['score_observation', 'metric_slope', '_score_and_group',
                                  '_score_or_group', 'score_group'])
@pytest.mark.parametrize('cls', [DefaultRanker, AdditiveRanker])
def test_shared_methods_live_only_on_the_base(cls, name):
    assert getattr(cls, name) is getattr(Ranker, name), (
        f'{cls.__name__}.{name} overrides the shared implementation; that is the '
        f'duplication GSCHED-1031 removed')


@pytest.mark.parametrize('cls', [DefaultRanker, AdditiveRanker])
def test_each_ranker_overrides_exactly_the_score_hook(cls):
    own = {n for n, v in vars(cls).items()
           if callable(v) and not n.startswith('__')}
    assert own == {'_combine_score_terms'}, f'{cls.__name__} should differ by one method only'


@pytest.mark.parametrize('module', [default_mod, additive_mod])
def test_the_parameter_classes_are_not_redefined_per_module(module):
    """additive.py used to carry its own RankerParameters copy.

    That made `default.RankerParameters is not additive.RankerParameters`, so AdditiveRanker
    only ever worked by duck typing on the params object engine/params.py built from
    default's class.
    """
    for name in ('RankerParameters', 'RankerBandParameters'):
        assert name not in vars(module), f'{module.__name__} redefines {name}'


def test_ranker_parameters_field_order_is_stable():
    """graphql_mid/schema.py:112 constructs RankerParameters positionally."""
    assert [f.name for f in fields(RankerParameters)][:6] == [
        'thesis_factor', 'power', 'met_power', 'vis_power', 'wha_power', 'air_power']


def test_altitude_limits_are_validated():
    with pytest.raises(ValueError, match='at least 18 degrees'):
        RankerParameters(gn_altitude_limits={MinMax.MIN: Angle(5.0 * u.deg),
                                             MinMax.MAX: Angle(88.0 * u.deg)})
    with pytest.raises(ValueError, match='90 degrees or less'):
        RankerParameters(gs_altitude_limits={MinMax.MIN: Angle(18.0 * u.deg),
                                             MinMax.MAX: Angle(91.0 * u.deg)})


@pytest.mark.skipif(not os.environ.get('REGENERATE_RANKER_GOLDENS'),
                    reason='set REGENERATE_RANKER_GOLDENS=1 to print fresh goldens')
def test_regenerate_goldens(collector):
    """Print replacements for the _GOLDEN_* constants. Not a check; always fails.

    Run it only when a scoring change is intended, and say so in the commit message.
    """
    lines = []
    for cls in (DefaultRanker, AdditiveRanker):
        digest = _score(cls, collector)[NIGHTS[0]].tobytes().hex()
        lines.append(f'_GOLDEN_{cls.__name__.replace("Ranker", "").upper()} = (')
        lines += [f"    '{digest[i:i + 64]}'" for i in range(0, len(digest), 64)]
        lines.append(')')
    pytest.fail('\n' + '\n'.join(lines), pytrace=False)
