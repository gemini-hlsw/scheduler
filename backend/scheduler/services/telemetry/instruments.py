# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""Instrument names, in one place because dashboards and alerts are written against them.
"""

__all__ = [
    'LOOP_STALL',
    'OPERATION_DURATION',
    'PROCESS_MEMORY_PEAK',
    'PROCESS_MEMORY_RSS',
    'SERVICE_NAME',
    'SIGHT_FALLBACK',
]

SERVICE_NAME = 'scheduler'

OPERATION_DURATION = 'scheduler.operation.duration'
PROCESS_MEMORY_RSS = 'scheduler.process.memory.rss'
PROCESS_MEMORY_PEAK = 'scheduler.process.memory.peak'
LOOP_STALL = 'scheduler.loop.stall'

# Sight visibility failed and the Collector computed locally instead. Without this the
# fallback reads as "Sight queries stopped and everything got slower", with no cause.
SIGHT_FALLBACK = 'scheduler.collector.sight_fallback'
