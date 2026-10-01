# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""Memory has to be readable without psutil, on Heroku and on a dev Mac.

The Heroku dyno is a single Python process, so this process's RSS is a usable proxy for
the number R14 fires on -- but only if the units are right. /proc reports kB and macOS's
getrusage reports bytes, and getting that wrong silently scales the alert threshold by
1024.
"""

import pytest

from scheduler.services.telemetry.memory import parse_proc_status, read_memory

PROC_STATUS = """\
Name:\tpython
Umask:\t0022
State:\tR (running)
Tgid:\t1
VmPeak:\t  2515432 kB
VmSize:\t  2481664 kB
VmHWM:\t   402100 kB
VmRSS:\t   318644 kB
Threads:\t9
"""


def test_rss_and_peak_are_parsed_as_bytes():
    reading = parse_proc_status(PROC_STATUS)

    assert reading.rss_bytes == 318644 * 1024
    assert reading.peak_bytes == 402100 * 1024


def test_a_status_file_without_the_fields_reads_as_unknown():
    reading = parse_proc_status('Name:\tpython\nState:\tR (running)\n')

    assert reading.rss_bytes is None
    assert reading.peak_bytes is None


def test_reading_memory_never_raises():
    # Called from the metric reader's own thread on every export; an exception there
    # kills the exporter silently and the memory alerts just stop firing.
    reading = read_memory()

    assert reading is not None


@pytest.mark.skipif(not __import__('sys').platform.startswith('linux'),
                    reason='/proc is Linux-only')
def test_linux_reports_a_live_rss():
    assert read_memory().rss_bytes > 0


def test_a_peak_is_always_available():
    # Linux has VmHWM, macOS has getrusage; between them peak is never unknown, which
    # is what makes the dev-box fallback worth having.
    assert read_memory().peak_bytes > 0
