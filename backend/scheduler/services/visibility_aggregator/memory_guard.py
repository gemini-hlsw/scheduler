# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import resource
import sys
from typing import Optional

from scheduler.config import config

__all__ = ["MemoryBudgetExceeded", "MemoryGuard", "process_memory_mb"]


class MemoryBudgetExceeded(RuntimeError):
    """Raised between units of work once the process is over its budget."""

    def __init__(self, phase: str, used_mb: float, budget_mb: float):
        self.phase = phase
        self.used_mb = used_mb
        self.budget_mb = budget_mb
        super().__init__(
            f"{used_mb:.0f} MB in use after {phase}, over the "
            f"{budget_mb:.0f} MB budget"
        )


def process_memory_mb() -> float:
    """Memory this process holds, counted the way Heroku counts it.

    Swap is included because once a dyno passes its quota the kernel pages it
    out: resident memory then plateaus while the dyno keeps getting heavier.
    Without /proc (macOS dev machines) this falls back to the peak resident
    size, which can only overstate current usage.
    """
    try:
        with open("/proc/self/status") as status:
            kb = sum(
                int(line.split()[1])
                for line in status
                if line.startswith(("VmRSS:", "VmSwap:"))
            )
        return kb / 1024
    except OSError:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is in bytes on macOS and kilobytes on Linux.
        return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


class MemoryGuard:
    """Checks process memory against a budget; ``None`` disables the guard."""

    def __init__(self, budget_mb: Optional[float]):
        self.budget_mb = None if budget_mb is None else float(budget_mb)

    @classmethod
    def from_config(cls) -> "MemoryGuard":
        return cls(getattr(config.visibility_aggregator, "memory_budget_mb", None))

    def check(self, phase: str) -> float:
        """Return memory in use (MB), raising once it is over budget.

        Call only after work is committed: the raise abandons whatever is
        still pending in the session.
        """
        used = process_memory_mb()
        if self.budget_mb is not None and used > self.budget_mb:
            raise MemoryBudgetExceeded(phase, used, self.budget_mb)
        return used
