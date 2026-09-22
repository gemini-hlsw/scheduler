# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""OpenTelemetry lifecycle: build the providers, flush them on the way out.

Everything that can be configured by environment variable is, so this module holds
almost no policy of its own. The scheduler runs on a Grafana Cloud stack shared with
other apps, so the two things it does insist on are the `service.name` identity and
keeping unbounded values out of anything that becomes a metric series.
"""

import os
from typing import Optional

from opentelemetry import metrics
from opentelemetry.metrics import Meter
from opentelemetry.sdk.metrics import (Counter, Histogram, MeterProvider,
                                       UpDownCounter)
from opentelemetry.sdk.metrics.export import (AggregationTemporality,
                                              PeriodicExportingMetricReader)
from opentelemetry.sdk.resources import Resource

from scheduler.services.logger_factory import create_logger
from scheduler.services.telemetry.instruments import SERVICE_NAME
from scheduler.services.telemetry.memory import register_memory_gauges
from scheduler.version import get_app_version

__all__ = ['current_provider', 'get_meter', 'is_sdk_disabled', 'process_type',
           'scheduler_mode', 'setup_telemetry', 'shutdown_telemetry']

_logger = create_logger(__name__, with_id=False)

# Observable gauges are always reported as-is; only the summing instruments need this.
_DELTA_TEMPORALITY = {
    Counter: AggregationTemporality.DELTA,
    Histogram: AggregationTemporality.DELTA,
    UpDownCounter: AggregationTemporality.DELTA,
}

_meter_provider: Optional[MeterProvider] = None


def is_sdk_disabled() -> bool:
    """Whether OTEL_SDK_DISABLED is set.

    Only the SDK's own autoconfiguration reads this variable. We build the providers by
    hand, so we have to honour it ourselves or the kill switch silently does nothing.
    """
    return os.environ.get('OTEL_SDK_DISABLED', '').strip().lower() == 'true'


def process_type() -> str:
    """``web`` from ``web.1``, ``run`` from ``run.1234``, ``local`` off Heroku.

    Heroku numbers one-off dynos with an unbounded counter, so the raw DYNO value as a
    metric attribute would mint a fresh series on every aggregator run and eat the
    shared stack's budget. The exact name still rides along on the perf event, where it
    is log metadata rather than a series key.
    """
    return os.environ.get('DYNO', '').split('.')[0] or 'local'


def scheduler_mode() -> str:
    """The deployment mode, straight from the environment.

    Deliberately not ``core.builder.modes.app_mode``: that module raises at import time
    when SCHEDULER_MODE is unset, and telemetry must never be the reason a script or a
    test fails to start.
    """
    return os.environ.get('SCHEDULER_MODE', '').strip().lower() or 'unknown'


def build_resource() -> Resource:
    """Identity attached to every signal.

    ``Resource.create`` folds in OTEL_RESOURCE_ATTRIBUTES, which is where deployment
    environment and namespace come from. The three set here win over the environment on
    purpose: the service name is how the shared stack tells our signals apart, the
    version is a build fact the deploy config has no business overriding, and the
    instance id has to be pinned (see below).

    ``service.instance.id`` defaults to a **random UUID per process**. Left alone, every
    aggregator dyno would arrive as a brand new instance and mint a fresh set of metric
    series on a stack other teams are paying for. Pinning it to the process type keeps
    it bounded; delta temporality (see `setup_telemetry`) is what makes several dynos
    sharing one id sum correctly instead of fighting over a cumulative counter.
    """
    return Resource.create({
        'service.name': SERVICE_NAME,
        'service.version': get_app_version(),
        'service.instance.id': process_type(),
    })


def current_provider() -> Optional[MeterProvider]:
    """The provider installed by `setup_telemetry`, if any."""
    return _meter_provider


def get_meter() -> Meter:
    """The scheduler's meter, or the API's no-op meter when setup never ran.

    Scripts and tests import the scheduler without configuring telemetry, and the no-op
    fallback is what keeps the instrumentation inert rather than fatal for them.
    """
    if _meter_provider is not None:
        return _meter_provider.get_meter(SERVICE_NAME)
    return metrics.get_meter(SERVICE_NAME)


def setup_telemetry(meter_provider: Optional[MeterProvider] = None,
                    set_global: bool = True) -> bool:
    """Configure metrics export and register the memory gauges.

    Args:
        meter_provider (MeterProvider): use this provider rather than building one from
            the environment. Tests pass an in-memory reader through here.
        set_global (bool): also install the provider as the process-wide default.
    Returns:
        True when telemetry is live, False when it is switched off.
    """
    global _meter_provider

    if meter_provider is None:
        if is_sdk_disabled():
            _logger.info('OTEL_SDK_DISABLED is set; telemetry export is off.')
            return False
        if not os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT'):
            _logger.info('No OTLP endpoint configured; telemetry export is off.')
            return False

        # Imported here so a deploy without an endpoint never pays for the exporter.
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import \
            OTLPMetricExporter

        # Delta, not the SDK default of cumulative. Most of our processes are one-off
        # aggregator dynos that live for one run: a cumulative counter from a process
        # that never comes back is a series that resets every time and reads as a
        # restart. Deltas from short-lived writers just add up, and Grafana Cloud's
        # gateway converts them back to cumulative on ingest, so queries are unchanged.
        reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(preferred_temporality=_DELTA_TEMPORALITY))
        meter_provider = MeterProvider(resource=build_resource(),
                                       metric_readers=[reader])

    _meter_provider = meter_provider
    if set_global:
        metrics.set_meter_provider(meter_provider)

    register_memory_gauges(get_meter(), {'process_type': process_type()})
    _logger.info(f'Telemetry configured for {SERVICE_NAME} ({process_type()}).')
    return True


def shutdown_telemetry(timeout_millis: int = 5000) -> None:
    """Flush and tear down the providers.

    Load-bearing on the aggregator's one-off dyno: a run can finish inside a whole
    export interval, and without this flush its metrics, including the `vis_agg.run`
    datapoint the alerts key on, are never sent at all.
    """
    global _meter_provider

    provider, _meter_provider = _meter_provider, None
    if provider is None:
        return

    try:
        provider.shutdown(timeout_millis=timeout_millis)
    except Exception as exc:  # pragma: no cover - exporter/network dependent
        _logger.warning(f'Telemetry shutdown did not complete cleanly: {exc}')
