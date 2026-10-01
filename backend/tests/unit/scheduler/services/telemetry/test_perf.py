# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""The timing façade must be safe enough to wrap the scheduler's hot paths.

Every one of these is a rule the call sites depend on: a measurement that survives an
exception (a run that dies is the one worth alerting on), a `.elapsed` that keeps the
existing human log lines intact, and a façade that cannot itself raise or block. The
instrument attributes are pinned here too, because they are what the Grafana alerts
query and renaming one silently breaks a dashboard rather than a test.
"""

import json
import logging

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from scheduler.context import schedule_id_var
from scheduler.services.telemetry import otel, perf
from scheduler.services.telemetry.perf import OPERATION_DURATION, perf_event, timed


@pytest.fixture
def reader():
    """A live meter provider the façade records into, torn down between tests."""
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    otel.setup_telemetry(meter_provider=provider, set_global=False)
    yield reader
    otel.shutdown_telemetry()


def points_for(reader, name=OPERATION_DURATION):
    """Flatten the reader's histogram data points for the given instrument."""
    data = reader.get_metrics_data()
    if data is None:
        return []
    return [point
            for resource_metric in data.resource_metrics
            for scope_metric in resource_metric.scope_metrics
            for metric in scope_metric.metrics
            if metric.name == name
            for point in metric.data.data_points]


def test_a_timed_block_records_one_measurement(reader):
    with timed('engine.plan'):
        pass

    points = points_for(reader)
    assert len(points) == 1, f"expected one data point, got {points}"
    assert points[0].count == 1
    assert points[0].attributes['operation'] == 'engine.plan'
    assert points[0].attributes['ok'] is True


def test_elapsed_is_available_to_the_caller(reader):
    # The existing human log lines interpolate this; losing it would mean rewriting
    # every message at the ~40 retrofit sites.
    with timed('collector.load') as t:
        pass

    assert t.elapsed >= 0.0


def test_a_failing_block_is_still_measured_and_reraises(reader):
    with pytest.raises(ValueError):
        with timed('vis_agg.run'):
            raise ValueError('boom')

    points = points_for(reader)
    assert len(points) == 1
    assert points[0].attributes['ok'] is False


def test_the_exception_class_stays_out_of_the_metric(reader):
    # Exception names are an open set; as a metric attribute each one would be another
    # series on a stack shared with other teams. It belongs on the event instead.
    with pytest.raises(ValueError):
        with timed('vis_agg.run'):
            raise ValueError('boom')

    assert 'error' not in points_for(reader)[0].attributes


def test_a_cancelled_block_is_measured(reader):
    # A killed Heroku dyno ends a run by cancellation, which is exactly the case the
    # absence alert needs to see recorded.
    import asyncio

    with pytest.raises(asyncio.CancelledError):
        with timed('vis_agg.run'):
            raise asyncio.CancelledError()

    assert points_for(reader)[0].attributes['ok'] is False


def test_operations_are_separate_series(reader):
    with timed('engine.plan'):
        pass
    with timed('collector.load'):
        pass

    operations = {point.attributes['operation'] for point in points_for(reader)}
    assert operations == {'engine.plan', 'collector.load'}


def test_instruments_follow_the_current_provider(reader):
    # The instrument is cached, so it has to be rebuilt when a provider is installed or
    # torn down. A stale one keeps recording into a dead provider and the metrics just
    # quietly stop arriving -- the worst kind of telemetry bug, since the alerts go
    # silent rather than red.
    with timed('engine.plan'):
        pass
    assert len(points_for(reader)) == 1

    second = InMemoryMetricReader()
    otel.setup_telemetry(meter_provider=MeterProvider(metric_readers=[second]),
                         set_global=False)
    with timed('engine.plan'):
        pass

    assert len(points_for(second)) == 1, 'recording did not follow the new provider'


def test_the_facade_is_inert_without_setup():
    # Scripts and tests import the scheduler without ever calling setup_telemetry;
    # the API's no-op meter must absorb that rather than blowing up at the call site.
    with timed('engine.plan') as t:
        pass

    assert t.elapsed >= 0.0
    perf_event('process.memory', rss_bytes=1)


def test_setup_is_skipped_when_the_sdk_is_disabled(monkeypatch):
    monkeypatch.setenv('OTEL_SDK_DISABLED', 'true')

    assert otel.setup_telemetry() is False
    assert otel.shutdown_telemetry() is None


class Capture(logging.Handler):
    """Collects the perf logger's records; it does not propagate to root, so caplog
    cannot see it."""

    def __init__(self):
        super().__init__()
        self.payloads = []

    def emit(self, record):
        self.payloads.append(json.loads(record.getMessage()))


@pytest.fixture
def emitted():
    handler = Capture()
    logger = logging.getLogger(perf.PERF_LOGGER_NAME)
    logger.addHandler(handler)
    yield handler.payloads
    logger.removeHandler(handler)


def test_an_event_is_emitted_as_one_json_line(emitted):
    perf_event('vis_agg.stage2_night', nights=3)

    assert len(emitted) == 1
    assert emitted[0]['event'] == 'vis_agg.stage2_night'
    assert emitted[0]['nights'] == 3


def test_a_timed_block_emits_its_duration(emitted):
    with timed('engine.plan'):
        pass

    assert emitted[0]['event'] == 'engine.plan'
    assert emitted[0]['duration_s'] >= 0.0
    assert emitted[0]['ok'] is True


def test_a_failing_block_names_the_exception_on_the_event(emitted):
    with pytest.raises(ValueError):
        with timed('vis_agg.run'):
            raise ValueError('boom')

    assert emitted[0]['ok'] is False
    assert emitted[0]['error'] == 'ValueError'


def test_the_run_id_rides_along_for_correlation(emitted):
    token = schedule_id_var.set('run-abc')
    try:
        perf_event('engine.plan')
    finally:
        schedule_id_var.reset(token)

    assert emitted[0]['run_id'] == 'run-abc'


def test_the_missing_run_id_sentinel_is_dropped(emitted):
    # schedule_id_var defaults to '3RR0R-Missing-ID'; shipping that on every line from
    # every background process would be noise in Loki, not a correlation key.
    perf_event('process.memory')

    assert 'run_id' not in emitted[0]


def test_an_unserialisable_field_does_not_break_the_caller(emitted):
    # Instrumentation must never be the thing that takes down a run.
    perf_event('engine.plan', obj=object())

    assert len(emitted) == 1


@pytest.mark.parametrize('dyno, expected', [
    ('web.1', 'web'),
    ('run.1234', 'run'),          # Heroku one-off dynos; the number is unbounded
    ('scheduler.1', 'scheduler'),
    ('', 'local'),
])
def test_the_process_type_drops_the_dyno_number(monkeypatch, dyno, expected):
    # The bare dyno name would mint a fresh metric series per aggregator run and eat
    # the shared Grafana stack's budget.
    monkeypatch.setenv('DYNO', dyno)

    assert otel.process_type() == expected


def test_the_process_type_is_local_off_heroku(monkeypatch):
    monkeypatch.delenv('DYNO', raising=False)

    assert otel.process_type() == 'local'


def test_the_instance_id_is_bounded(monkeypatch):
    # The SDK's default is a fresh random UUID per process. Left alone, every one-off
    # aggregator dyno would arrive as a new instance and mint a whole new set of series
    # on a Grafana stack shared with other teams.
    monkeypatch.setenv('DYNO', 'run.4711')

    assert otel.build_resource().attributes['service.instance.id'] == 'run'


def test_the_service_identity_survives_a_hostile_env(monkeypatch):
    # service.name is how the shared stack separates our signals from everyone else's,
    # so a stray OTEL_RESOURCE_ATTRIBUTES must not be able to rename us.
    monkeypatch.setenv('OTEL_RESOURCE_ATTRIBUTES',
                       'service.name=something-else,deployment.environment=staging')
    attributes = otel.build_resource().attributes

    assert attributes['service.name'] == 'scheduler'
    assert attributes['deployment.environment'] == 'staging'
