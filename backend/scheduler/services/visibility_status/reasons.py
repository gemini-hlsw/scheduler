# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause


from dataclasses import dataclass
from typing import Optional

__all__ = [
    "NON_SIDEREAL",
    "NO_SITE",
    "UNSUPPORTED_TARGET",
    "NO_COORDINATES",
    "ODB_CHANGED",
    "NIGHT_NOT_COMPUTED",
    "TARGET_NOT_IN_SIGHT",
    "STAGE1_MISSING",
    "PROBABLY_PARSER_ERROR",
    "UNKNOWN",
    "TERMINAL_REASONS",
    "SightNightState",
    "missing_reason",
]

# --- terminal: the observation cannot be stored -------------------------------

# The aggregator parses non-sidereal targets but does not compute them.
NON_SIDEREAL = "NON_SIDEREAL"
# The instrument does not resolve to GN or GS, so there is no site to compute at.
NO_SITE = "NO_SITE"
# The asterism entry is neither sidereal nor non-sidereal, which the target
# mapping cannot build and the aggregator's parse raises on.
UNSUPPORTED_TARGET = "UNSUPPORTED_TARGET"
# A sidereal target whose coordinates are absent in the ODB.
NO_COORDINATES = "NO_COORDINATES"

TERMINAL_REASONS = frozenset(
    {NON_SIDEREAL, NO_SITE, UNSUPPORTED_TARGET, NO_COORDINATES}
)

# --- stored, but no longer current -------------------------------------------

# The ODB reports its visibility inputs changed after the last successful run.
ODB_CHANGED = "ODB_CHANGED"

# --- progress: a real subject with no row yet, earliest stage first -----------

# No night_events row for its site on this night: the aggregator has not reached
# this night at all, so nothing per-target could exist either.
NIGHT_NOT_COMPUTED = "NIGHT_NOT_COMPUTED"
# The target is not in the Sight DB. The aggregator only creates targets it
# parsed, so this covers a parse failure (one bad observation loses its whole
# program), a target no run has created yet, and one renamed in the ODB, since
# Sight keys targets by name.
TARGET_NOT_IN_SIGHT = "TARGET_NOT_IN_SIGHT"
# The target exists but has no Stage-1 positions for this site and night.
STAGE1_MISSING = "STAGE1_MISSING"
# Everything upstream is in place and only Stage 2 is absent. The likeliest
# cause is a parse failure on the observation itself: targets are never deleted,
# so one created by an earlier run keeps its Stage-1 positions after the
# observation using it stops parsing, and a single bad observation takes its
# whole program's remaining observations with it (aggregator
# ``_collect_requests``). A run in progress looks the same from here, hence
# "probably"; the aggregator card on the same screen shows whether one is going.
PROBABLY_PARSER_ERROR = "PROBABLY_PARSER_ERROR"
# The Sight DB could not be read, so none of the checks above ran.
UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class SightNightState:
    """How far the Sight DB has got for one night, as membership sets.

    Membership rather than rows: the Stage-1 tables carry per-minute arrays, so
    answering "is it there?" by loading rows pulls hundreds of megabytes for a
    semester's targets, where sets of ids are a few thousand entries.
    """

    # Site keys ("GN" / "GS") that have night events computed for the night.
    nights_computed: frozenset[str] = frozenset()
    # Target names present in the Sight DB.
    targets_known: frozenset[str] = frozenset()
    # (target name, site key) pairs with Stage-1 positions for the night.
    stage1_ready: frozenset[tuple[str, str]] = frozenset()


def missing_reason(
    target_name: Optional[str],
    site: Optional[str],
    state: Optional[SightNightState],
) -> str:
    """Why this observation has no Stage-2 row, in pipeline order.

    Walks the chain the aggregator builds — night events, target, Stage-1
    positions, Stage 2 — and names the first link that is absent, so one
    uncomputed night explains all of its observations at once. The parser is
    blamed only once every link is intact.

    ``state`` is None when the Sight reads failed: no link was checked, so the
    answer is ``UNKNOWN``.
    """
    if state is None:
        return UNKNOWN
    if site is None or site not in state.nights_computed:
        return NIGHT_NOT_COMPUTED
    if target_name is None or target_name not in state.targets_known:
        return TARGET_NOT_IN_SIGHT
    if (target_name, site) not in state.stage1_ready:
        return STAGE1_MISSING
    return PROBABLY_PARSER_ERROR
