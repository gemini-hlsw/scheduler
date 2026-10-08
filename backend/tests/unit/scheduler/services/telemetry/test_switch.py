# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""`telemetry.enabled` (TELEMETRY_ENABLED) turns the metrics export and the perf events off.

Off must stay off for both outputs, but `timed` has to keep measuring and keep raising:
the call sites read `.elapsed` for their own log lines, and a block that fails must still
fail. An explicit meter provider is a request for telemetry and wins over the switch.
"""

import logging

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from scheduler.services.telemetry import otel, perf
from scheduler.services.telemetry.perf import perf_event, timed

from .test_perf import Capture, points_for


@pytest.fixture
def switched_off(monkeypatch):
    """Telemetry with the config switch off, restored from the real config afterwards."""
    monkeypatch.setenv('TELEMETRY_ENABLED', 'false')
    otel.shutdown_telemetry()
    yield
    monkeypatch.delenv('TELEMETRY_ENABLED')
    otel.shutdown_telemetry()


@pytest.fixture
def emitted():
    handler = Capture()
    logger = logging.getLogger(perf.PERF_LOGGER_NAME)
    logger.addHandler(handler)
    yield handler.payloads
    logger.removeHandler(handler)


def test_the_switch_defaults_to_on(monkeypatch):
    monkeypatch.delenv('TELEMETRY_ENABLED', raising=False)
    assert otel.is_enabled_by_config() is True


def test_the_env_var_turns_it_off(monkeypatch):
    monkeypatch.setenv('TELEMETRY_ENABLED', 'false')
    assert otel.is_enabled_by_config() is False


def test_off_emits_no_perf_events(switched_off, emitted):
    perf_event('vis_agg.stage2_night', nights=3)
    with timed('engine.plan'):
        pass

    assert emitted == []


def test_off_still_measures_elapsed(switched_off):
    with timed('engine.plan') as t:
        pass

    assert t.elapsed >= 0.0


def test_off_still_raises_from_the_block(switched_off):
    with pytest.raises(ValueError, match='boom'):
        with timed('engine.plan'):
            raise ValueError('boom')


def test_off_installs_no_exporter_even_with_an_endpoint(switched_off, monkeypatch):
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://localhost:4318')

    assert otel.setup_telemetry() is False
    assert otel.telemetry.enabled is False


def test_an_explicit_provider_overrides_the_switch(switched_off, emitted):
    reader = InMemoryMetricReader()
    otel.setup_telemetry(meter_provider=MeterProvider(metric_readers=[reader]), set_global=False)

    with timed('engine.plan'):
        pass

    assert len(points_for(reader)) == 1
    assert [e['event'] for e in emitted] == ['engine.plan']


def test_shutdown_returns_to_the_config_switch(switched_off, emitted):
    otel.setup_telemetry(meter_provider=MeterProvider(metric_readers=[InMemoryMetricReader()]),
                         set_global=False)
    otel.shutdown_telemetry()

    perf_event('vis_agg.stage2_night')

    assert emitted == []
