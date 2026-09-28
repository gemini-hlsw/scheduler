# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import os
from typing import Optional

from opentelemetry import metrics
from opentelemetry.metrics import Histogram, Meter
from opentelemetry.sdk.metrics import Counter, MeterProvider, UpDownCounter
from opentelemetry.sdk.metrics import Histogram as SDKHistogram
from opentelemetry.sdk.metrics.export import (AggregationTemporality,
                                              PeriodicExportingMetricReader)
from opentelemetry.sdk.resources import Resource

from scheduler.services.logger_factory import create_logger
from scheduler.services.telemetry.instruments import OPERATION_DURATION, SERVICE_NAME
from scheduler.services.telemetry.memory import register_memory_gauges
from scheduler.version import get_app_version

__all__ = ['Telemetry', 'build_resource', 'is_sdk_disabled', 'process_type',
           'scheduler_mode', 'setup_telemetry', 'shutdown_telemetry', 'telemetry']

_logger = create_logger(__name__, with_id=False)

# Observable gauges are always reported as-is; only the summing instruments need this.
_DELTA_TEMPORALITY = {
    Counter: AggregationTemporality.DELTA,
    SDKHistogram: AggregationTemporality.DELTA,
    UpDownCounter: AggregationTemporality.DELTA,
}


def is_sdk_disabled() -> bool:
    """Whether OTEL_SDK_DISABLED is set.

    Only the SDK's own autoconfiguration reads this variable. We build the providers by
    hand, so we have to honour it ourselves or the kill switch silently does nothing.
    """
    return os.environ.get('OTEL_SDK_DISABLED', '').strip().lower() == 'true'


def process_type() -> str:
    """Grabs the id type for process run in Heroku (``web`` or ``id``) and strip the number id given so
    each run is not saved separately and depletes the stack budget in Grafana.
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


def _build_meter_provider() -> MeterProvider:
    """A provider wired to the OTLP exporter described by the environment."""
    # Imported here so a deploy without an endpoint never pays for the exporter.
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

    # Delta, not the SDK default of cumulative. Most of our processes are one-off
    # aggregator dynos that live for one run: a cumulative counter from a process that
    # never comes back is a series that resets every time and reads as a restart.
    # Deltas from short-lived writers just add up, and Grafana Cloud's gateway converts
    # them back to cumulative on ingest, so queries are unchanged.
    reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(preferred_temporality=_DELTA_TEMPORALITY))

    return MeterProvider(resource=build_resource(), metric_readers=[reader])


class Telemetry:
    """Owns the meter provider and the instruments built from it.

    One instance per process, exported as `telemetry` below. It is ambient
    infrastructure in the same way logging is: the ~40 measurement sites are spread
    across the call graph and threading a handle to all of them would cost more than it
    buys. What this does buy over loose module state is a single object whose lifecycle
    is explicit, and instruments that cannot outlive the provider they came from.
    """

    def __init__(self) -> None:
        self._provider: Optional[MeterProvider] = None
        # Built eagerly, and rebuilt on every configure/shutdown. A lazy cache would
        # need a lock: `timed` is called from worker threads via asyncio.to_thread.
        self._operation_duration = self._build_operation_duration()

    @property
    def enabled(self) -> bool:
        """Whether a provider is installed; False means the instruments are no-ops."""
        return self._provider is not None

    @property
    def meter(self) -> Meter:
        """This process's meter, or the API's no-op meter when setup never ran.

        Scripts and tests import the scheduler without configuring telemetry, and the
        no-op fallback is what keeps instrumentation inert rather than fatal for them.
        """
        if self._provider is not None:
            return self._provider.get_meter(SERVICE_NAME)
        return metrics.get_meter(SERVICE_NAME)

    @property
    def operation_duration(self) -> Histogram:
        """The duration histogram every `timed` block records into."""
        return self._operation_duration

    def _build_operation_duration(self) -> Histogram:
        return self.meter.create_histogram(
            OPERATION_DURATION,
            unit='s',
            description='Wall time of a named scheduler operation.')

    def configure(self,
                  meter_provider: Optional[MeterProvider] = None,
                  set_global: bool = True) -> bool:
        """Install a provider, rebuild the instruments, register the memory gauges.

        Args:
            meter_provider (MeterProvider): use this provider rather than building one
                from the environment. Tests pass an in-memory reader through here.
            set_global (bool): also install it as the process-wide default.
        Returns:
            True when telemetry is live, False when it is switched off.
        """
        if meter_provider is None:
            if is_sdk_disabled():
                _logger.info('OTEL_SDK_DISABLED is set; telemetry export is off.')
                return False
            if not os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT'):
                _logger.info('No OTLP endpoint configured; telemetry export is off.')
                return False
            meter_provider = _build_meter_provider()

        self._provider = meter_provider
        if set_global:
            metrics.set_meter_provider(meter_provider)

        self._operation_duration = self._build_operation_duration()
        register_memory_gauges(self.meter, {'process_type': process_type()})
        _logger.info(f'Telemetry configured for {SERVICE_NAME} ({process_type()}).')
        return True

    def shutdown(self, timeout_millis: int = 5000) -> None:
        """Flush and tear down the provider.

        Load-bearing on the aggregator's one-off dyno: a run can finish inside a whole
        export interval, and without this flush its metrics, including the
        `vis_agg.run` datapoint the alerts key on, are never sent at all.
        """
        provider, self._provider = self._provider, None
        self._operation_duration = self._build_operation_duration()
        if provider is None:
            return

        try:
            provider.shutdown(timeout_millis=timeout_millis)
        except Exception as exc:  # pragma: no cover - exporter/network dependent
            _logger.warning(f'Telemetry shutdown did not complete cleanly: {exc}')


telemetry = Telemetry()


def setup_telemetry(meter_provider: Optional[MeterProvider] = None,
                    set_global: bool = True) -> bool:
    """Configure this process's telemetry. See `Telemetry.configure`."""
    return telemetry.configure(meter_provider=meter_provider, set_global=set_global)


def shutdown_telemetry(timeout_millis: int = 5000) -> None:
    """Flush and tear down this process's telemetry. See `Telemetry.shutdown`."""
    telemetry.shutdown(timeout_millis=timeout_millis)
