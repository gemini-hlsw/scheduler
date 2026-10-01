# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""Performance telemetry: durations and process memory, exported over OTLP.

Call sites only need `timed` and `perf_event`. Process entry points additionally call
`setup_telemetry` / `shutdown_telemetry`.
"""

from scheduler.services.telemetry.memory import MemoryReading, read_memory
from scheduler.services.telemetry.otel import setup_telemetry, shutdown_telemetry
from scheduler.services.telemetry.perf import Timing, perf_event, timed

__all__ = ['MemoryReading', 'Timing', 'perf_event', 'read_memory', 'setup_telemetry',
           'shutdown_telemetry', 'timed']
