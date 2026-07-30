# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""Which observations *should* have visibility stored, read live from the ODB.

This is the "expected" half of the coverage answer; the "stored" half comes from
``visibility_data``. It is deliberately uncached: the scheduler runs in real
time and the architecture depends on acting on the latest ODB state, so a
snapshot table or a TTL here would reintroduce exactly the staleness the design
rejects. The cost is a paginated sweep per call.

For the same reason nothing here degrades quietly. A failed page raises, because
a silently truncated sweep is indistinguishable from "nothing is missing" — the
precise false negative the Visibility tab exists to catch.
"""

from dataclasses import dataclass
from typing import Optional

from gpp_client.generated.input_types import (
    WhereCalculatedObservationWorkflow,
    WhereObservation,
    WhereOrderObservationWorkflowState,
    WhereOrderProgramId,
    WhereProgram,
)

from scheduler.clients.gpp import gpp
from scheduler.night_monitor.event_handlers.obscalc_visibility import (
    site_key_from_instrument,
)
from scheduler.services import logger_factory
from scheduler.services.visibility_aggregator.aggregator import SCHEDULABLE_STATES

_logger = logger_factory.create_logger(__name__)

__all__ = [
    "ExpectedObservation",
    "get_expected_observations",
    "SKIP_NON_SIDEREAL",
    "SKIP_NO_TARGET",
    "SKIP_NO_SITE",
]

# Why an observation can never appear in visibility_data. Surfaced in the UI so
# a permanent gap reads as "not applicable" rather than "broken".
SKIP_NON_SIDEREAL = "NON_SIDEREAL"
SKIP_NO_TARGET = "NO_TARGET"
SKIP_NO_SITE = "NO_SITE"

# Observations per ODB page. Large enough to keep the round-trip count down on a
# few-thousand-observation semester, small enough that one response stays a
# sane size.
_PAGE_SIZE = 500


@dataclass(frozen=True)
class ExpectedObservation:
    """One observation the Sight DB is expected to hold visibility for.

    ``observation_id`` is the reference label (``G-…``) — the same key
    ``visibility_data.observation_id`` uses, and the only id shown in the UI.
    ``internal_id`` is the ODB GID (``o-…``), kept solely to match the
    visibility-changes feed; it is never exposed over GraphQL.
    """

    observation_id: str
    internal_id: str
    program_id: str
    program_label: str
    site: Optional[str]
    target_name: Optional[str]
    is_sidereal: bool
    skip_reason: Optional[str]


def _classify(match, program_label: str) -> Optional[ExpectedObservation]:
    """Build an ExpectedObservation, or None when there is nothing to key on.

    An observation without a reference label cannot be matched against
    ``visibility_data`` in either direction, so it is dropped rather than
    reported as missing.
    """
    reference = getattr(match, "reference", None)
    label = getattr(reference, "label", None) if reference is not None else None
    if not label:
        _logger.debug(f"Observation {match.id} has no reference label; skipping.")
        return None

    asterism = getattr(match.target_environment, "asterism", None) or []
    base = asterism[0] if asterism else None

    target_name = str(base.name) if base is not None and base.name else None
    is_sidereal = base is not None and base.sidereal is not None
    site = site_key_from_instrument(match.instrument)

    if base is None or target_name is None:
        skip_reason = SKIP_NO_TARGET
    elif not is_sidereal:
        skip_reason = SKIP_NON_SIDEREAL
    elif site is None:
        skip_reason = SKIP_NO_SITE
    else:
        skip_reason = None

    return ExpectedObservation(
        observation_id=str(label),
        internal_id=str(match.id),
        program_id=str(match.program.id),
        program_label=program_label,
        site=site,
        target_name=target_name,
        is_sidereal=is_sidereal,
        skip_reason=skip_reason,
    )


async def get_expected_observations() -> list[ExpectedObservation]:
    """Every schedulable observation of the currently available programs.

    Two live ODB reads: the available-programs list, then a paginated sweep of
    READY/ONGOING observations belonging to them. Raises if any page fails or if
    pagination stops making progress — never returns a partial set.
    """
    programs = await gpp.client.scheduler.get_all_reference_labels()
    labels_by_program_id = {
        str(program_id): str(label) for label, program_id in programs
    }
    if not labels_by_program_id:
        _logger.info("No available programs; expected set is empty.")
        return []

    where = WhereObservation(
        program=WhereProgram(
            id=WhereOrderProgramId(in_=list(labels_by_program_id))
        ),
        workflow=WhereCalculatedObservationWorkflow(
            workflow_state=WhereOrderObservationWorkflowState(
                in_=SCHEDULABLE_STATES
            )
        ),
    )

    expected: list[ExpectedObservation] = []
    seen: set[str] = set()
    offset: Optional[str] = None
    pages = 0

    while True:
        response = await gpp.client.observation.get_all(
            where=where, offset=offset, limit=_PAGE_SIZE
        )
        pages += 1
        matches = response.observations.matches
        # The ODB cursor is inclusive of the offset id, so a page repeats the
        # previous page's last row. Dedupe by GID rather than assuming either
        # convention.
        fresh = [m for m in matches if str(m.id) not in seen]
        for match in fresh:
            seen.add(str(match.id))
            observation = _classify(
                match, labels_by_program_id.get(str(match.program.id), "")
            )
            if observation is not None:
                expected.append(observation)

        if not response.observations.has_more:
            break
        if not fresh:
            # has_more is set but the page added nothing new: without this the
            # loop would spin forever.
            raise RuntimeError(
                f"ODB pagination stopped making progress after {pages} pages "
                f"({len(seen)} observations, offset={offset!r}); aborting rather "
                f"than looping."
            )
        offset = str(matches[-1].id)

    _logger.info(
        f"Expected set: {len(expected)} observations across "
        f"{len(labels_by_program_id)} programs in {pages} ODB pages."
    )
    return expected
