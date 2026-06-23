# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpp_client.generated.enums import (
    Instrument,
    ObservationWorkflowState,
    SkyBackground,
    TimingWindowInclusion,
)

from scheduler.night_monitor.event_handlers import obscalc_visibility as ov
from scheduler.night_monitor.event_handlers.odb_event_handler import ODBEventHandler
from scheduler.services.sight.calculator.models import ElevationType


# --- builders for fake obscalc event values -------------------------------------------------

def _sidereal(hms="05:35:16.8", dms="-05:23:24.0", epoch="J2000.000"):
    # Coords come from the sexagesimal hms/dms strings (the degrees fields are
    # buggy for negative dec). Defaults are M42: RA ~83.82 deg, Dec ~-5.39 deg.
    return SimpleNamespace(
        ra=SimpleNamespace(hms=hms, hours=None, degrees=None),
        dec=SimpleNamespace(dms=dms, degrees=None),
        epoch=epoch,
    )


def _asterism(name, sidereal=None, nonsidereal=None):
    return SimpleNamespace(name=name, sidereal=sidereal, nonsidereal=nonsidereal)


def _airmass(lo, hi):
    return SimpleNamespace(min=lo, max=hi)


def _hourangle(lo, hi):
    return SimpleNamespace(min_hours=lo, max_hours=hi)


def _end_at(at_utc):
    return SimpleNamespace(at_utc=at_utc)


def _end_after(seconds, period_seconds=None, times=None):
    repeat = None
    if period_seconds is not None:
        repeat = SimpleNamespace(period=SimpleNamespace(seconds=period_seconds), times=times)
    return SimpleNamespace(after=SimpleNamespace(seconds=seconds), repeat=repeat)


def _tw(start_utc, end, inclusion=TimingWindowInclusion.INCLUDE):
    return SimpleNamespace(inclusion=inclusion, start_utc=start_utc, end=end)


def _value(asterism=None, air_mass=None, hour_angle=None, sb=SkyBackground.DARK,
           timing_windows=None, start="2025-02-01", end="2025-02-05",
           state=ObservationWorkflowState.READY, obs_id="o-123"):
    return SimpleNamespace(
        id=obs_id,
        workflow=SimpleNamespace(value=SimpleNamespace(state=state)),
        target_environment=SimpleNamespace(asterism=asterism or [], explicit_base=None),
        constraint_set=SimpleNamespace(
            sky_background=sb,
            elevation_range=SimpleNamespace(air_mass=air_mass, hour_angle=hour_angle),
        ),
        timing_windows=timing_windows or [],
        program=SimpleNamespace(active=SimpleNamespace(start=start, end=end)),
    )


_RANGE_END = datetime(2025, 2, 5, tzinfo=timezone.utc)


# --- site_key_from_instrument ---------------------------------------------------------------

@pytest.mark.parametrize("instrument,expected", [
    (Instrument.GMOS_NORTH, "GN"),    # NORTH suffix
    (Instrument.GMOS_SOUTH, "GS"),    # SOUTH suffix
    (Instrument.ACQ_CAM_NORTH, "GN"),  # suffix
    (Instrument.FLAMINGOS2, "GS"),    # explicit map
    (Instrument.GNIRS, "GN"),         # explicit map
    (Instrument.GHOST, "GS"),         # explicit map
    (None, None),
])
def test_site_key_from_instrument(instrument, expected):
    assert ov.site_key_from_instrument(instrument) == expected


# --- build_target_create --------------------------------------------------------------------

def test_build_target_create_sidereal():
    value = _value(asterism=[_asterism("M42", sidereal=_sidereal())])
    payload = ov.build_target_create(value)
    assert payload is not None
    assert payload.name == "M42"
    assert payload.is_sidereal is True
    assert payload.base_ra == pytest.approx(83.82, abs=1e-2)
    assert payload.base_dec == pytest.approx(-5.39, abs=1e-2)
    assert payload.epoch == pytest.approx(2000.0)


def test_build_target_create_negative_dec():
    # Dec parsed from dms must stay in [-90, 90] even though GPP's buggy
    # dec.degrees would report ~327 for a -33 deg declination.
    value = _value(asterism=[_asterism("X", sidereal=_sidereal(dms="-33:03:00.0"))])
    payload = ov.build_target_create(value)
    assert payload.base_dec == pytest.approx(-33.05, abs=1e-2)


@pytest.mark.parametrize("raw,expected", [
    ("J2000.000", 2000.0),
    ("B1950.000", 1950.0),
    (2000.0, 2000.0),
    ("2015.5", 2015.5),
    (None, 2000.0),
    ("garbage", 2000.0),
])
def test_parse_epoch(raw, expected):
    assert ov._parse_epoch(raw) == pytest.approx(expected)


def test_build_target_create_nonsidereal_skipped():
    value = _value(asterism=[_asterism("Ceres", sidereal=None,
                                       nonsidereal=SimpleNamespace(des="1"))])
    assert ov.build_target_create(value) is None


def test_build_target_create_no_asterism():
    assert ov.build_target_create(_value(asterism=[])) is None


# --- build_constraints ----------------------------------------------------------------------

def test_build_constraints_airmass():
    value = _value(air_mass=_airmass(1.0, 1.6), sb=SkyBackground.DARK)
    c = ov.build_constraints(value, _RANGE_END)
    assert c.elevation_type == ElevationType.AIRMASS
    assert c.elevation_min == pytest.approx(1.0)
    assert c.elevation_max == pytest.approx(1.6)
    assert c.target_sb == pytest.approx(0.5)
    assert c.has_resources and c.can_schedule


def test_build_constraints_hour_angle():
    value = _value(air_mass=None, hour_angle=_hourangle(-4.0, 4.0))
    c = ov.build_constraints(value, _RANGE_END)
    assert c.elevation_type == ElevationType.HOUR_ANGLE
    assert c.elevation_min == pytest.approx(-4.0)
    assert c.elevation_max == pytest.approx(4.0)


def test_build_constraints_default_elevation():
    value = _value(air_mass=None, hour_angle=None)
    c = ov.build_constraints(value, _RANGE_END)
    assert c.elevation_type == ElevationType.AIRMASS
    assert c.elevation_min == pytest.approx(1.0)
    assert c.elevation_max == pytest.approx(2.0)


@pytest.mark.parametrize("sb,frac", [
    (SkyBackground.DARKEST, 0.2),
    (SkyBackground.DARK, 0.5),
    (SkyBackground.GRAY, 0.8),
    (SkyBackground.BRIGHT, 1.0),
])
def test_sky_background_mapping(sb, frac):
    c = ov.build_constraints(_value(air_mass=_airmass(1.0, 2.0), sb=sb), _RANGE_END)
    assert c.target_sb == pytest.approx(frac)


# --- expand_event_timing_windows ------------------------------------------------------------

def test_expand_end_at():
    out = ov.expand_event_timing_windows(
        [_tw("2025-02-01 00:00:00", _end_at("2025-02-02 00:00:00"))], _RANGE_END
    )
    assert len(out) == 1
    assert out[0].start == datetime(2025, 2, 1, tzinfo=timezone.utc)
    assert out[0].end == datetime(2025, 2, 2, tzinfo=timezone.utc)


def test_expand_end_after_non_repeating():
    out = ov.expand_event_timing_windows(
        [_tw("2025-02-01 00:00:00", _end_after(3600))], _RANGE_END
    )
    assert len(out) == 1
    assert (out[0].end - out[0].start).total_seconds() == pytest.approx(3600)


def test_expand_end_after_finite_repeats():
    # one day duration, repeat every day, 2 extra repeats => 3 windows
    out = ov.expand_event_timing_windows(
        [_tw("2025-02-01 00:00:00", _end_after(3600, period_seconds=86400, times=2))],
        _RANGE_END,
    )
    assert len(out) == 3
    assert [w.start.day for w in out] == [1, 2, 3]


def test_expand_end_after_forever_capped_by_range_end():
    out = ov.expand_event_timing_windows(
        [_tw("2025-02-01 00:00:00", _end_after(3600, period_seconds=86400, times=None))],
        _RANGE_END,
    )
    # windows on 02-01..02-05 (start <= range_end), then stop
    assert [w.start.day for w in out] == [1, 2, 3, 4, 5]


def test_expand_open_ended_capped_by_range_end():
    out = ov.expand_event_timing_windows(
        [_tw("2025-02-01 00:00:00", None)], _RANGE_END
    )
    assert len(out) == 1
    assert out[0].end == _RANGE_END


def test_expand_skips_exclude():
    out = ov.expand_event_timing_windows(
        [_tw("2025-02-01 00:00:00", _end_at("2025-02-02 00:00:00"),
             inclusion=TimingWindowInclusion.EXCLUDE)],
        _RANGE_END,
    )
    assert out == []


# --- calculate_and_store_visibility orchestrator --------------------------------------------

@pytest.mark.asyncio
async def test_calculate_and_store_visibility_invokes_calculator():
    value = _value(
        asterism=[_asterism("M42", sidereal=_sidereal())],
        air_mass=_airmass(1.0, 2.0),
    )
    mock_calc = MagicMock()
    # New target: get_by_name returns None, so create is called.
    mock_calc.target_repo.get_by_name = AsyncMock(return_value=None)
    mock_calc.target_repo.create = AsyncMock()
    mock_calc.store_missing_visibility = AsyncMock(
        return_value={"stored": 5, "nights": 5, "already_present": 0}
    )

    @asynccontextmanager
    async def fake_scope():
        yield MagicMock()

    with patch.object(ov, "session_scope", fake_scope), \
         patch.object(ov, "Calculator", return_value=mock_calc):
        result = await ov.calculate_and_store_visibility(
            value, observation_id="GS-2025A-Q-101-23", site_key="GS"
        )

    mock_calc.target_repo.create.assert_awaited_once()
    mock_calc.store_missing_visibility.assert_awaited_once()
    req_list = mock_calc.store_missing_visibility.await_args.args[0]
    assert req_list[0].observation_id == "GS-2025A-Q-101-23"
    assert req_list[0].site_id == "GS"
    assert req_list[0].target_name == "M42"
    assert result["stored"] == 5
    assert result["target_existed"] is False
    assert "elapsed_seconds" in result


@pytest.mark.asyncio
async def test_calculate_and_store_visibility_skips_create_for_existing_target():
    value = _value(
        asterism=[_asterism("M42", sidereal=_sidereal())],
        air_mass=_airmass(1.0, 2.0),
    )
    mock_calc = MagicMock()
    # Existing target: get_by_name returns a target, so create must NOT be called.
    mock_calc.target_repo.get_by_name = AsyncMock(return_value=MagicMock())
    mock_calc.target_repo.create = AsyncMock()
    mock_calc.store_missing_visibility = AsyncMock(
        return_value={"stored": 0, "nights": 5, "already_present": 5}
    )

    @asynccontextmanager
    async def fake_scope():
        yield MagicMock()

    with patch.object(ov, "session_scope", fake_scope), \
         patch.object(ov, "Calculator", return_value=mock_calc):
        result = await ov.calculate_and_store_visibility(
            value, observation_id="GS-2025A-Q-101-23", site_key="GS"
        )

    mock_calc.target_repo.create.assert_not_called()
    assert result["target_existed"] is True
    assert result["already_present"] == 5


@pytest.mark.asyncio
async def test_calculate_and_store_visibility_skips_non_sidereal():
    value = _value(asterism=[_asterism("Ceres", sidereal=None)])
    with patch.object(ov, "Calculator") as mock_calc_cls:
        result = await ov.calculate_and_store_visibility(value, "GS-1", "GS")
    mock_calc_cls.assert_not_called()
    assert result["stored"] == 0


# --- _on_updated_edit handler ---------------------------------------------------------------

def _handler():
    return ODBEventHandler(scheduler_queue=AsyncMock())


def _obs_response(label, instrument=Instrument.GMOS_SOUTH):
    ref = SimpleNamespace(label=label) if label is not None else None
    return SimpleNamespace(
        observation=SimpleNamespace(reference=ref, instrument=instrument)
    )


@pytest.mark.asyncio
async def test_on_updated_edit_ready_computes_and_replans():
    handler = _handler()
    value = _value(state=ObservationWorkflowState.READY, obs_id="o-1")
    event = SimpleNamespace(value=value, edit_type="UPDATED")

    mock_gpp = MagicMock()
    mock_gpp.client.observation.get_by_id = AsyncMock(
        return_value=_obs_response("G-2026A-0397-D-0067", Instrument.GMOS_SOUTH)
    )

    with patch("scheduler.night_monitor.event_handlers.odb_event_handler.gpp", mock_gpp), \
         patch("scheduler.night_monitor.event_handlers.odb_event_handler."
               "sight_visibility_enabled", return_value=True), \
         patch("scheduler.night_monitor.event_handlers.odb_event_handler."
               "calculate_and_store_visibility", new_callable=AsyncMock) as mock_calc:
        await handler._on_updated_edit(event)

    mock_gpp.client.observation.get_by_id.assert_awaited_once_with("o-1")
    mock_calc.assert_awaited_once()
    assert mock_calc.await_args.kwargs == {
        "observation_id": "G-2026A-0397-D-0067",
        "site_key": "GS",
    }
    handler.scheduler_queue.add_schedule_event.assert_awaited_once()


@pytest.mark.asyncio
async def test_on_updated_edit_not_ready_is_noop():
    handler = _handler()
    value = _value(state=ObservationWorkflowState.DEFINED)
    event = SimpleNamespace(value=value, edit_type="UPDATED")

    with patch("scheduler.night_monitor.event_handlers.odb_event_handler.gpp") as mock_gpp, \
         patch("scheduler.night_monitor.event_handlers.odb_event_handler."
               "calculate_and_store_visibility", new_callable=AsyncMock) as mock_calc:
        await handler._on_updated_edit(event)

    mock_gpp.client.observation.get_by_id.assert_not_called()
    mock_calc.assert_not_called()
    handler.scheduler_queue.add_schedule_event.assert_not_called()


@pytest.mark.asyncio
async def test_on_updated_edit_unresolvable_site_still_replans():
    handler = _handler()
    value = _value(state=ObservationWorkflowState.READY, obs_id="o-2")
    event = SimpleNamespace(value=value, edit_type="UPDATED")

    mock_gpp = MagicMock()
    # Reference label present but instrument missing -> site unresolved.
    mock_gpp.client.observation.get_by_id = AsyncMock(
        return_value=_obs_response("G-2026A-0397-D-0067", instrument=None)
    )

    with patch("scheduler.night_monitor.event_handlers.odb_event_handler.gpp", mock_gpp), \
         patch("scheduler.night_monitor.event_handlers.odb_event_handler."
               "sight_visibility_enabled", return_value=True), \
         patch("scheduler.night_monitor.event_handlers.odb_event_handler."
               "calculate_and_store_visibility", new_callable=AsyncMock) as mock_calc:
        await handler._on_updated_edit(event)

    mock_calc.assert_not_called()
    handler.scheduler_queue.add_schedule_event.assert_awaited_once()


@pytest.mark.asyncio
async def test_on_updated_edit_local_strategy_skips_sight():
    """When the visibility strategy is local, the handler must not touch GPP/sight
    but must still trigger a replan."""
    handler = _handler()
    value = _value(state=ObservationWorkflowState.READY, obs_id="o-3")
    event = SimpleNamespace(value=value, edit_type="UPDATED")

    with patch("scheduler.night_monitor.event_handlers.odb_event_handler.gpp") as mock_gpp, \
         patch("scheduler.night_monitor.event_handlers.odb_event_handler."
               "sight_visibility_enabled", return_value=False), \
         patch("scheduler.night_monitor.event_handlers.odb_event_handler."
               "calculate_and_store_visibility", new_callable=AsyncMock) as mock_calc:
        await handler._on_updated_edit(event)

    mock_gpp.client.observation.get_by_id.assert_not_called()
    mock_calc.assert_not_called()
    handler.scheduler_queue.add_schedule_event.assert_awaited_once()
