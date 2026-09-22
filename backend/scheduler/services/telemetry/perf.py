# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""The timing façade the ~40 measurement sites call.

Two outputs from one call. The histogram is what the Grafana alerts query; the JSON
event is what you read afterwards to find out why a run was slow. `.elapsed` exists so
the existing human log lines keep working untouched -- the aggregator's ETA messages are
genuinely useful to someone watching a backfill, and this must not cost them.

Nothing here may raise. It wraps the scheduler's hot paths, so a bug in instrumentation
would be an outage in scheduling.
"""

import json
import logging
import os
import socket
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from opentelemetry.metrics import Histogram

from scheduler.context import schedule_id_var
from scheduler.services.telemetry import otel
from scheduler.services.telemetry.instruments import OPERATION_DURATION
from scheduler.version import get_app_version

__all__ = ['OPERATION_DURATION', 'PERF_LOGGER_NAME', 'Timing', 'perf_event', 'timed']

PERF_LOGGER_NAME = 'scheduler.perf'

# The default of scheduler.context.schedule_id_var. It means "no run in scope" rather
# than naming one, so it is dropped instead of shipped on every background line.
_MISSING_RUN_ID = '3RR0R-Missing-ID'

_VERSION = get_app_version()
_HOSTNAME = socket.gethostname()

_histogram: Optional[Histogram] = None
_histogram_provider = None


def _build_perf_logger() -> logging.Logger:
    """A stream of one JSON object per line, separate from the human log stream."""
    logger = logging.getLogger(PERF_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    # The message is already the whole record; root's formatter would wrap it in prose.
    logger.propagate = False
    return logger


_perf_logger = _build_perf_logger()


class Timing:
    """Handle yielded by `timed`. `.elapsed` feeds the caller's own log line."""
    __slots__ = ('elapsed',)

    def __init__(self) -> None:
        self.elapsed = 0.0


def _operation_histogram() -> Histogram:
    """The duration histogram, rebuilt whenever the meter provider changes."""
    global _histogram, _histogram_provider

    provider = otel.current_provider()
    if _histogram is None or provider is not _histogram_provider:
        _histogram = otel.get_meter().create_histogram(
            OPERATION_DURATION,
            unit='s',
            description='Wall time of a named scheduler operation.')
        _histogram_provider = provider

    return _histogram


def _base_fields() -> Dict[str, Any]:
    """Context stamped on every event. Cheap: the expensive lookups are cached."""
    fields: Dict[str, Any] = {
        'dyno': os.environ.get('DYNO') or _HOSTNAME,
        'process_type': otel.process_type(),
        'mode': otel.scheduler_mode(),
        'version': _VERSION,
    }

    run_id = schedule_id_var.get()
    if run_id and run_id != _MISSING_RUN_ID:
        fields['run_id'] = run_id

    return fields


def perf_event(event: str, **fields: Any) -> None:
    """Emit one JSON line describing `event`.

    Args:
        event (str): the operation or sample name, e.g. ``vis_agg.stage2_night``.
        **fields: extra context. Free to be high-cardinality -- these are log
            attributes, not metric attributes, so a run id or a night is fine here.
    """
    try:
        payload = {'event': event, **_base_fields(), **fields}
        # default=repr: a caller passing something unserialisable should get a slightly
        # worse log line, never an exception in the middle of a scheduling run.
        _perf_logger.info(json.dumps(payload, default=repr))
    except Exception:  # pragma: no cover - the façade must not break its caller
        pass


@contextmanager
def timed(operation: str, **fields: Any) -> Iterator[Timing]:
    """Time a block, record the histogram, and emit the matching event.

    Args:
        operation (str): one of the names in the plan's operation list. Keep the set
            small; every value is a metric series on a shared stack.
        **fields: extra context for the JSON event only.
    Yields:
        a `Timing` whose `.elapsed` is set once the block completes.
    """
    timing = Timing()
    started = time.perf_counter()
    error: Optional[str] = None

    try:
        yield timing
    except BaseException as exc:
        # BaseException, not Exception: an asyncio cancellation is how a killed dyno
        # ends a run, and that is precisely the case worth measuring.
        error = type(exc).__name__
        raise
    finally:
        timing.elapsed = time.perf_counter() - started
        ok = error is None

        try:
            # `error` stays out of the metric attributes on purpose: exception class
            # names are an open set, and each one would be another series.
            _operation_histogram().record(timing.elapsed,
                                          {'operation': operation, 'ok': ok})
        except Exception:  # pragma: no cover - defensive
            pass

        event_fields = dict(fields)
        if error is not None:
            event_fields['error'] = error
        perf_event(operation,
                   duration_s=round(timing.elapsed, 6),
                   ok=ok,
                   **event_fields)
