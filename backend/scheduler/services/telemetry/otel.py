# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import os
from typing import Optional

from opentelemetry import metrics
from opentelemetry.metrics import Counter as APICounter
from opentelemetry.metrics import Histogram, Meter
from opentelemetry.sdk.metrics import Counter, MeterProvider, UpDownCounter
from opentelemetry.sdk.metrics import Histogram as SDKHistogram
from opentelemetry.sdk.metrics.export import (AggregationTemporality,
                                              PeriodicExportingMetricReader)
from opentelemetry.sdk.resources import Resource

from scheduler.services.logger_factory import create_logger
from scheduler.services.telemetry.instruments import (LOOP_STALL, OPERATION_DURATION,
                                                      SERVICE_NAME, SIGHT_FALLBACK)
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
    """Whether OTEL_SDK_DISABLED is set."""
    return os.environ.get('OTEL_SDK_DISABLED', '').strip().lower() == 'true'


def process_type() -> str:
    """Grabs the id type for process run in Heroku (``web`` or ``id``) and strip the number id given so
    each run is not saved separately and depletes the stack budget in Grafana.
    """
    return os.environ.get('DYNO', '').split('.')[0] or 'local'


def scheduler_mode() -> str:
    """The deployment mode, straight from the environment."""
    return os.environ.get('SCHEDULER_MODE', '').strip().lower() or 'unknown'


def build_resource() -> Resource:
    """Identity attached to every signal."""
    return Resource.create({
        'service.name': SERVICE_NAME,
        'service.version': get_app_version(),
        'service.instance.id': process_type(),
    })


def _build_meter_provider() -> MeterProvider:
    """A provider wired to the OTLP exporter described by the environment."""
    # Imported here so a deploy without an endpoint never pays for the exporter.
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

    reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(preferred_temporality=_DELTA_TEMPORALITY))

    return MeterProvider(resource=build_resource(), metric_readers=[reader])


class Telemetry:
    """Owns the meter provider and the instruments built from it.
    """

    def __init__(self) -> None:
        self._provider: Optional[MeterProvider] = None
        # Built eagerly, and rebuilt on every configure/shutdown. A lazy cache would
        # need a lock: `timed` is called from worker threads via asyncio.to_thread.
        self._build_instruments()

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

    @property
    def loop_stall(self) -> Histogram:
        """Event-loop stalls, recorded only past the monitor's warning threshold."""
        return self._loop_stall

    @property
    def sight_fallback(self) -> APICounter:
        """Times the Collector fell back from Sight to local visibility computation."""
        return self._sight_fallback

    def _build_instruments(self) -> None:
        meter = self.meter
        self._operation_duration = meter.create_histogram(
            OPERATION_DURATION,
            unit='s',
            description='Wall time of a named scheduler operation.')
        self._loop_stall = meter.create_histogram(
            LOOP_STALL,
            unit='s',
            description='Time the event loop could not give a waiting task.')
        self._sight_fallback = meter.create_counter(
            SIGHT_FALLBACK,
            description='Sight visibility loads that fell back to local computation.')

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

        self._build_instruments()
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
        self._build_instruments()
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
