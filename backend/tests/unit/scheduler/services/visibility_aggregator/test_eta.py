# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""
A run reports where it is and when it expects to finish through the heartbeat
`detail` payload, which runner.py commits to the scheduler_coordination row and
the Visibility tab reads back.

The ETA is a moving estimate: measured throughput so far extrapolated over the
work remaining. It deliberately reports None rather than a guess when nothing
has been measured yet, so the UI can say "estimating" instead of showing a
fabricated number.

The payload is stored as JSONB (coordination._heartbeat json.dumps it), so every
value has to be JSON-native.
"""
import json

from scheduler.services.visibility_aggregator.aggregator import (
    _format_duration,
    progress_detail,
    progress_eta_seconds,
)


# --- ETA math ---------------------------------------------------------------

def test_eta_extrapolates_from_measured_throughput():
    # 10s for 2 of 10 units -> 5 s/unit, 8 left -> 40s.
    assert progress_eta_seconds(elapsed_seconds=10.0, done=2, total=10) == 40.0


def test_eta_is_unknown_before_anything_is_measured():
    # Nothing done yet: any extrapolation would be invented.
    assert progress_eta_seconds(elapsed_seconds=3.0, done=0, total=10) is None


def test_eta_is_zero_once_the_work_is_complete():
    assert progress_eta_seconds(elapsed_seconds=50.0, done=10, total=10) == 0.0


def test_eta_is_zero_when_more_was_done_than_planned():
    assert progress_eta_seconds(elapsed_seconds=50.0, done=11, total=10) == 0.0


def test_eta_is_unknown_when_there_is_no_work():
    # An empty semester / no targets must not divide by zero.
    assert progress_eta_seconds(elapsed_seconds=1.0, done=0, total=0) is None


# --- duration formatting ----------------------------------------------------

def test_format_duration_uses_seconds_below_ninety():
    assert _format_duration(45.0) == "45s"
    assert _format_duration(89.4) == "89s"


def test_format_duration_uses_minutes_up_to_ninety_minutes():
    assert _format_duration(90.0) == "1.5m"
    assert _format_duration(5399.0) == "90.0m"


def test_format_duration_uses_hours_beyond_ninety_minutes():
    assert _format_duration(5400.0) == "1.5h"


# --- heartbeat payload ------------------------------------------------------

def test_progress_detail_carries_phase_progress_and_eta():
    detail = progress_detail(
        "stage1",
        done=25,
        total=100,
        unit="targets",
        elapsed_seconds=50.0,
        started_at="2026-07-29T12:00:00+00:00",
    )

    assert detail["phase"] == "stage1"
    assert detail["progress_current"] == 25
    assert detail["progress_total"] == 100
    assert detail["progress_unit"] == "targets"
    assert detail["elapsed_seconds"] == 50.0
    # 50s for 25 of 100 -> 2 s/target, 75 left -> 150s.
    assert detail["eta_seconds"] == 150.0
    assert detail["started_at"] == "2026-07-29T12:00:00+00:00"


def test_progress_detail_merges_phase_specific_extras():
    detail = progress_detail(
        "stage2",
        done=3,
        total=180,
        unit="nights",
        elapsed_seconds=30.0,
        night="2026-08-01",
        stored=412,
    )

    assert detail["night"] == "2026-08-01"
    assert detail["stored"] == 412
    assert detail["progress_unit"] == "nights"


def test_progress_detail_reports_unknown_eta_as_null():
    detail = progress_detail(
        "stage1", done=0, total=100, unit="targets", elapsed_seconds=2.0
    )

    assert detail["eta_seconds"] is None


def test_progress_detail_is_json_serializable():
    # coordination._heartbeat json.dumps this straight into a JSONB column.
    detail = progress_detail(
        "stage2", done=3, total=180, unit="nights", elapsed_seconds=30.123456
    )

    assert json.loads(json.dumps(detail)) == detail


def test_progress_detail_rounds_seconds_for_readability():
    detail = progress_detail(
        "stage2", done=3, total=180, unit="nights", elapsed_seconds=30.123456
    )

    assert detail["elapsed_seconds"] == 30.1
    assert detail["eta_seconds"] == round(30.123456 / 3 * 177, 1)
