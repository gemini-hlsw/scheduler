# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""Instrument names, in one place because dashboards and alerts are written against them.

The `scheduler.` prefix is what separates us from the other apps on the shared Grafana
Cloud stack. Mimir's Prometheus translation rewrites the dots and appends the unit, so
`scheduler.operation.duration` is queried as `scheduler_operation_duration_seconds`.
Renaming any of these breaks a dashboard, not a test.
"""

__all__ = [
    'LOOP_STALL',
    'OPERATION_DURATION',
    'PROCESS_MEMORY_PEAK',
    'PROCESS_MEMORY_RSS',
    'SERVICE_NAME',
]

SERVICE_NAME = 'scheduler'

OPERATION_DURATION = 'scheduler.operation.duration'
PROCESS_MEMORY_RSS = 'scheduler.process.memory.rss'
PROCESS_MEMORY_PEAK = 'scheduler.process.memory.peak'
LOOP_STALL = 'scheduler.loop.stall'
