# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause


import asyncio
import functools
import json
import logging
import os
import socket
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional, TypeVar

from scheduler.context import schedule_id_var
from scheduler.services.telemetry import otel
from scheduler.services.telemetry.instruments import OPERATION_DURATION
from scheduler.version import get_app_version

__all__ = ['OPERATION_DURATION', 'PERF_LOGGER_NAME', 'Timing', 'perf_event',
           'record_loop_stall', 'record_sight_fallback', 'timed', 'timed_function']

F = TypeVar('F', bound=Callable[..., Any])

PERF_LOGGER_NAME = 'scheduler.perf'

# The default of scheduler.context.schedule_id_var. It means "no run in scope" rather
# than naming one, so it is dropped instead of shipped on every background line.
_MISSING_RUN_ID = '3RR0R-Missing-ID'

_VERSION = get_app_version()
_HOSTNAME = socket.gethostname()


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
    if not otel.telemetry.emitting:
        return
    try:
        payload = {'event': event, **_base_fields(), **fields}
        # default=repr: a caller passing something unserialisable should get a slightly
        # worse log line, never an exception in the middle of a scheduling run.
        _perf_logger.info(json.dumps(payload, default=repr))
    except Exception:  # pragma: no cover - the façade must not break its caller
        pass


def timed_function(operation: str, **fields: Any) -> Callable[[F], F]:
    """Decorator form of `timed`, for when the whole function body is the operation.

    Use it instead of a `with` block when wrapping would mean re-indenting a long body,
    or when a method has several call sites that would each need their own block.

    Handles async and sync alike: decorating a coroutine function with the plain `with`
    form would time only how long it took to *create* the coroutine, which is the kind
    of measurement that looks fine and means nothing.
    """
    def decorate(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with timed(operation, **fields):
                    return await func(*args, **kwargs)
            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with timed(operation, **fields):
                return func(*args, **kwargs)
        return wrapper  # type: ignore[return-value]

    return decorate


def record_sight_fallback() -> None:
    """Count one Collector fallback from Sight to local visibility computation."""
    try:
        otel.telemetry.sight_fallback.add(1)
    except Exception:  # pragma: no cover - defensive
        pass


def record_loop_stall(seconds: float) -> None:
    """Record an event-loop stall.

    Only for stalls past the monitor's warning threshold. Recording every sample would
    be ten datapoints a second of "the loop is fine", which buries the real ones.
    """
    try:
        otel.telemetry.loop_stall.record(seconds)
    except Exception:  # pragma: no cover - defensive
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
        # No `return` here: inside `finally` it would swallow the block's exception.
        if otel.telemetry.emitting:
            try:
                # `error` stays out of the metric attributes on purpose: exception class
                # names are an open set, and each one would be another series.
                otel.telemetry.operation_duration.record(timing.elapsed,
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
