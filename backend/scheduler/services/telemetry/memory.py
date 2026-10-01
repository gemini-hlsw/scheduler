# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import resource
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from opentelemetry.metrics import CallbackOptions, Meter, Observation

from scheduler.services.telemetry.instruments import (PROCESS_MEMORY_PEAK,
                                                      PROCESS_MEMORY_RSS)

__all__ = ['MemoryReading', 'parse_proc_status', 'read_memory', 'register_memory_gauges']

_PROC_STATUS = '/proc/self/status'
_KB = 1024


@dataclass(frozen=True)
class MemoryReading:
    """Current and high-water RSS in bytes; ``None`` where the platform cannot say."""
    rss_bytes: Optional[int]
    peak_bytes: Optional[int]


def parse_proc_status(text: str) -> MemoryReading:
    """Pull VmRSS and VmHWM out of a ``/proc/<pid>/status`` body.

    Args:
        text (str): the file's contents.
    Returns:
        the reading, in bytes; /proc reports kB, hence the scaling.
    """
    found: Dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(':')
        if key in ('VmRSS', 'VmHWM'):
            parts = rest.split()
            if parts and parts[0].isdigit():
                found[key] = int(parts[0]) * _KB

    return MemoryReading(rss_bytes=found.get('VmRSS'), peak_bytes=found.get('VmHWM'))


def _rusage_peak_bytes() -> Optional[int]:
    """High-water RSS from getrusage.

    ``ru_maxrss`` is kB on Linux but bytes on macOS. Getting that wrong silently scales
    the alert threshold by 1024, so the platform check is not optional.
    """
    try:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (OSError, ValueError):  # pragma: no cover - platform dependent
        return None

    return peak if sys.platform == 'darwin' else peak * _KB


def read_memory() -> MemoryReading:
    """This process's memory use.

    Never raises: this runs on the metric reader's export thread, where an exception
    kills the exporter quietly and the memory alerts simply stop firing.
    """
    try:
        with open(_PROC_STATUS, 'r') as status:
            reading = parse_proc_status(status.read())
        if reading.rss_bytes is not None:
            return reading
    except OSError:
        pass

    # No /proc, i.e. a dev Mac. getrusage knows the high-water mark but not the current
    # value, and inventing one from the peak would quietly lie to the dashboard.
    return MemoryReading(rss_bytes=None, peak_bytes=_rusage_peak_bytes())


def register_memory_gauges(meter: Meter,
                           attributes: Optional[Dict[str, Any]] = None) -> None:
    """Attach the RSS and peak observable gauges to ``meter``.

    Args:
        meter (Meter): the meter to register on.
        attributes (dict): attributes stamped on every observation, e.g. process_type.
    """
    stamped = attributes or {}

    def observe_rss(options: CallbackOptions) -> Iterable[Observation]:
        rss_bytes = read_memory().rss_bytes
        if rss_bytes is not None:
            yield Observation(rss_bytes, stamped)

    def observe_peak(options: CallbackOptions) -> Iterable[Observation]:
        peak_bytes = read_memory().peak_bytes
        if peak_bytes is not None:
            yield Observation(peak_bytes, stamped)

    meter.create_observable_gauge(
        PROCESS_MEMORY_RSS,
        callbacks=[observe_rss],
        unit='By',
        description='Resident set size of this scheduler process.')
    meter.create_observable_gauge(
        PROCESS_MEMORY_PEAK,
        callbacks=[observe_peak],
        unit='By',
        description='High-water resident set size of this scheduler process.')
