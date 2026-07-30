# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
import time
from datetime import date, datetime, timedelta, timezone
from typing import Awaitable, Callable, NamedTuple, Optional

import numpy as np
from astropy.time import Time
from lucupy import sky
from lucupy.minimodel import ALL_SITES, ObservationClass
from lucupy.minimodel.semester import Semester
from lucupy.types import ZeroTime
from gpp_client.rest.models import VisibilityChanges
from gpp_client.generated.input_types import (
    ObservationWorkflowState,
    WhereCalculatedObservationWorkflow,
    WhereObservation,
    WhereOrderObservationId,
    WhereOrderObservationWorkflowState,
    WhereOrderTargetId,
    WhereTarget,
)
from sqlalchemy.ext.asyncio import AsyncSession

from scheduler.config import config
from scheduler.clients.gpp import gpp
from scheduler.core.programprovider.gpp import GppProgramProvider, gpp_program_data
from scheduler.core.sources.sources import Sources
from scheduler.services import logger_factory
from scheduler.services.visibility_aggregator.coordination import (
    get_change_watermark,
    set_change_watermark,
)
from scheduler.services.sight._temporary.lucupy_adapters import (
    stage2_constraints,
    target_create,
)
from scheduler.services.sight.calculator.calculator import Calculator
from scheduler.services.sight.calculator.models import ObservationRequest

_logger = logger_factory.create_logger(__name__)

__all__ = [
    "run_aggregation",
    "is_night_in_progress",
    "progress_detail",
    "progress_eta_seconds",
    "SCHEDULABLE_STATES",
    "resolve_schedulable_observation_labels",
    "LabelResolution",
]

# Heartbeat callback: an async fn the runner passes so it can refresh its
# coordination row (and surface progress) between chunks of work.
Heartbeat = Callable[[dict], Awaitable[None]]

_OBS_CLASSES = frozenset({
    ObservationClass.SCIENCE,
    ObservationClass.PROGCAL,
    ObservationClass.PARTNERCAL,
})

# Only these observations are schedulable, so only these are worth refreshing.
# Same states gpp-client's SchedulerDomain.get_all filters the program dump to,
# which is why every observation in this run's parse set is already one of them.
SCHEDULABLE_STATES = [
    ObservationWorkflowState.READY,
    ObservationWorkflowState.ONGOING,
]

# Backoff (seconds) between visibility-changes fetch attempts. A transient ODB
# blip should not cost a whole Heroku Scheduler cycle (the next tick is an hour
# away); a persistent outage still fails the run after the last attempt.
_CHANGES_FETCH_BACKOFFS = (5.0, 15.0, 30.0)

# Changed internal ids resolved per ODB query. `limit` is set to the chunk
# size, so a chunk always fits in one page and no pagination is needed.
_ODB_ID_CHUNK = 100


def _as_utc_datetime(t: Time) -> datetime:
    """Astropy Time to a UTC datetime."""
    return np.atleast_1d(t.to_datetime(timezone=timezone.utc))[0]


def _format_duration(seconds: float) -> str:
    """Human-readable duration for ETA logging."""
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def progress_eta_seconds(
    *, elapsed_seconds: float, done: int, total: int
) -> Optional[float]:
    """Seconds remaining, extrapolated from throughput measured so far.

    Returns None when there is nothing to extrapolate from — no units finished
    yet, or no work at all — so callers can report "estimating" rather than a
    fabricated number. Returns 0.0 once the work is complete.
    """
    if total <= 0 or done <= 0:
        return None
    if done >= total:
        return 0.0
    return elapsed_seconds / done * (total - done)


def progress_detail(
    phase: str,
    *,
    done: int,
    total: int,
    unit: str,
    elapsed_seconds: float,
    **extra,
) -> dict:
    """Heartbeat payload describing where the run is and when it will finish.

    ``runner.py`` commits this to the ``scheduler_coordination`` row, where the
    Visibility tab reads it back, so every value has to be JSON-native — hence
    the rounding rather than raw floats.
    """
    eta = progress_eta_seconds(
        elapsed_seconds=elapsed_seconds, done=done, total=total
    )
    return {
        "phase": phase,
        "progress_current": done,
        "progress_total": total,
        "progress_unit": unit,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "eta_seconds": None if eta is None else round(eta, 1),
        **extra,
    }


def is_night_in_progress(now: Optional[datetime] = None) -> bool:
    """True if it is currently dark time (12 deg twilight) at GN or GS.

    We check both the night anchored to ``now`` and to ``now - 1 day`` so a night whose twilights
    are dated to the previous calendar day is still detected.
    """
    now = now or datetime.now(timezone.utc)
    for site in ALL_SITES:
        for day_offset in (0, -1):
            try:
                events = sky.night_events(
                    Time(now + timedelta(days=day_offset)),
                    site.location,
                    site.timezone,
                )
                eve_twi = _as_utc_datetime(events[3])
                morn_twi = _as_utc_datetime(events[4])
            except Exception as exc:
                _logger.warning(f"night_events failed for {site.name}: {exc}")
                continue
            if eve_twi <= now <= morn_twi:
                _logger.info(
                    f"Dark time in progress at {site.name} "
                    f"({eve_twi.isoformat()} .. {morn_twi.isoformat()})."
                )
                return True
    return False


async def _collect_requests(program_ids: list[str], range_end: datetime):
    """Parse the available GPP programs into sidereal targets + obs requests.

    Also returns a mapping of ODB observation internal id (``o-…``, what the
    visibility-changes endpoint reports) to the reference label Sight stores
    as ``observation_id``.
    """
    provider = GppProgramProvider(_OBS_CLASSES, Sources())

    targets_by_name: dict = {}
    requests: list[ObservationRequest] = []
    labels_by_internal_id: dict[str, str] = {}
    skipped_no_target = 0
    skipped_nonsidereal = 0
    bad_programs = 0

    program_data = await gpp_program_data(program_ids)
    processed = 0
    async for raw in program_data:
        processed += 1
        # Parsing is CPU-bound and the async-generator body has no awaits, so a
        # large semester-start dump would otherwise block the event loop (and
        # starve the runner's heartbeat task). Yield periodically so heartbeats
        # keep firing and the coordination row stays fresh.
        if processed % 25 == 0:
            await asyncio.sleep(0)
        try:
            data = next(iter(raw.values())) if len(raw.keys()) == 1 else raw
            program = provider.parse_program(data)
            if program is None:
                continue
            if program.program_awarded() == ZeroTime:
                continue

            for obs in program.observations():
                base = obs.base_target()
                # We are skipping no target observations as they should not be ready
                if base is None:
                    skipped_no_target += 1
                    continue
                payload = target_create(base)
                if payload is None:
                    skipped_no_target += 1
                    continue
                # TODO: Add NonSideral support
                if not payload.is_sidereal:
                    skipped_nonsidereal += 1
                    continue

                targets_by_name.setdefault(payload.name, payload)
                labels_by_internal_id[str(obs.internal_id)] = obs.id.id
                requests.append(ObservationRequest(
                    observation_id=obs.id.id,
                    target_name=str(base.name),
                    site_id=obs.site.name,
                    constraints=stage2_constraints(
                        obs,
                        has_resources=True,
                        can_schedule=True,
                        range_end=range_end,
                        program_start=program.start,
                        program_end=program.end,
                    ),
                ))
        except Exception as exc:
            bad_programs += 1
            _logger.warning(f"Failed to process a program: {exc}")

    if bad_programs:
        _logger.info(f"Skipped {bad_programs} unparseable programs.")
    counts = {
        "skipped_no_target": skipped_no_target,
        "skipped_nonsidereal": skipped_nonsidereal,
    }
    return targets_by_name, requests, labels_by_internal_id, counts


async def _load_changes(
    session: AsyncSession,
    now: datetime,
) -> tuple[VisibilityChanges, datetime]:
    """Fetch ODB entities whose visibility inputs changed since the watermark.

    ``since`` is the persisted watermark (advanced only after a successful
    run), falling back to ``now - changes_fallback_lookback_hours`` when the
    row does not exist yet. The fetch is retried with a short backoff
    (``_CHANGES_FETCH_BACKOFFS``) so a transient blip does not cost a whole
    hourly cycle; once the attempts are exhausted the failure propagates and
    fails the run, the watermark stays put, and the next run retries the same
    window.

    Returns the parsed changes plus the fetch time (captured before the HTTP
    call; the new watermark is derived from it, minus the safety overlap).
    """
    lookback = timedelta(
        hours=float(config.visibility_aggregator.changes_fallback_lookback_hours)
    )
    watermark = await get_change_watermark(session)
    since = watermark or (now - lookback)
    fetch_time = datetime.now(timezone.utc)

    attempts = len(_CHANGES_FETCH_BACKOFFS) + 1
    for attempt, backoff in enumerate(_CHANGES_FETCH_BACKOFFS + (None,), start=1):
        try:
            changes = await gpp.client.scheduler.get_visibility_changes(since)
            break
        except Exception as exc:
            if backoff is None:
                _logger.error(
                    f"Visibility-changes fetch failed after {attempts} attempts; "
                    f"failing the run (watermark untouched): {exc}"
                )
                raise
            _logger.warning(
                f"Visibility-changes fetch attempt {attempt}/{attempts} failed "
                f"({exc}); retrying in {backoff:.0f}s."
            )
            await asyncio.sleep(backoff)
    _logger.info(
        f"ODB visibility changes since {since.isoformat()}"
        f"{' (fallback window)' if watermark is None else ''}: "
        f"{len(changes.observation_ids)} observations, "
        f"{len(changes.target_ids)} targets."
    )
    return changes, fetch_time


async def _resolve_target_names(internal_ids: list[str]) -> dict[str, str]:
    """Map changed target internal ids to ODB names, one query per chunk.

    A failed chunk is logged and leaves its internal ids unresolved; the
    caller counts those as skipped rather than failing the run.
    """
    names: dict[str, str] = {}
    for start in range(0, len(internal_ids), _ODB_ID_CHUNK):
        chunk = internal_ids[start:start + _ODB_ID_CHUNK]
        try:
            response = await gpp.client.target.get_all(
                where=WhereTarget(id=WhereOrderTargetId(in_=chunk)),
                limit=len(chunk),
            )
        except Exception as exc:
            _logger.warning(
                f"Could not resolve a chunk of {len(chunk)} changed targets: {exc}"
            )
            continue
        for match in response.targets.matches:
            if match.name is not None:
                names[str(match.id)] = str(match.name)
    return names


class LabelResolution(NamedTuple):
    """Resolved labels plus how many chunks failed to come back.

    The aggregator ignores ``failed_chunks`` (a missed label just means one less
    invalidation this tick), but a read-only consumer needs to know its answer
    is incomplete rather than reporting an empty result as authoritative.
    """

    labels: dict[str, str]
    failed_chunks: int


async def resolve_schedulable_observation_labels(
    internal_ids: list[str],
) -> LabelResolution:
    """Map changed observation internal ids to labels, schedulable ones only.

    The visibility-changes endpoint reports every observation whose visibility
    inputs changed, regardless of workflow state, so the query filters to
    READY/ONGOING (``SCHEDULABLE_STATES``) — the same filter gpp-client
    applies to the program dump. Internal ids in another state simply do not
    come back and are left alone. One query per chunk.

    Sight stores the reference label as ``observation_id``, while the endpoint
    reports internal ids, hence the mapping.
    """
    labels: dict[str, str] = {}
    failed_chunks = 0
    for start in range(0, len(internal_ids), _ODB_ID_CHUNK):
        chunk = internal_ids[start:start + _ODB_ID_CHUNK]
        try:
            response = await gpp.client.observation.get_all(
                where=WhereObservation(
                    id=WhereOrderObservationId(in_=chunk),
                    workflow=WhereCalculatedObservationWorkflow(
                        workflow_state=WhereOrderObservationWorkflowState(
                            in_=SCHEDULABLE_STATES
                        )
                    ),
                ),
                limit=len(chunk),
            )
        except Exception as exc:
            failed_chunks += 1
            _logger.warning(
                f"Could not resolve a chunk of {len(chunk)} changed "
                f"observations: {exc}"
            )
            continue
        for match in response.observations.matches:
            reference = match.reference
            if reference is not None and reference.label:
                labels[str(match.id)] = str(reference.label)
    return LabelResolution(labels=labels, failed_chunks=failed_chunks)


async def _apply_odb_changes(
    calc: Calculator,
    changes: VisibilityChanges,
    targets_by_name: dict,
    requests: list[ObservationRequest],
    labels_by_internal_id: dict[str, str],
) -> dict:
    """Refresh Sight rows for the entities the ODB reported as changed.

    Changed targets: overwrite the stored coordinates with this run's freshly
    parsed values (bumping ``updated_at``, so the Stage-1 batch loop that runs
    next recomputes their night positions) and invalidate Stage 2 for every
    observation of that target.

    Changed observations: Stage 2 only — delete their visibility rows so
    ``_store_missing_visibility`` recomputes them with the fresh constraints.
    Stage 1 is untouched since the target did not move.

    Both sets are filtered to schedulable work before anything is written: the
    endpoint reports changes for observations in any workflow state, so
    non-READY/ONGOING ones are dropped by the resolving query, and a changed
    target only counts if it is in this run's parse set (itself built from
    READY/ONGOING observations only).

    Per-item failures are logged and skipped; deletions are committed once at
    the end. Everything here is idempotent, so re-applying an overlap window
    is safe.
    """
    labels_by_target: dict[str, list[str]] = {}
    for request in requests:
        labels_by_target.setdefault(request.target_name, []).append(
            request.observation_id
        )

    labels_to_invalidate: set[str] = set()
    targets_updated = 0
    # Skips are counted per reason: most are benign (a change the aggregator
    # has no work for), so a single total would hide the ones that are not.
    targets_unresolved = 0
    targets_not_in_parse_set = 0
    targets_pending_create = 0
    targets_update_failed = 0
    obs_skipped = 0

    # Resolve every changed target internal id in batched ODB queries, then
    # load the matching Sight rows in a single query, so the cost stays flat
    # as the change set grows.
    names_by_internal_id = await _resolve_target_names(sorted(changes.target_ids))
    internal_ids_by_name: dict[str, list[str]] = {}
    for internal_id in sorted(changes.target_ids):
        name = names_by_internal_id.get(internal_id)
        if name is None:
            # The ODB did not return it: deleted (the query excludes deleted
            # targets), or its chunk query failed and warned above.
            _logger.info(
                f"Changed target {internal_id} could not be resolved in the "
                f"ODB; skipping."
            )
            targets_unresolved += 1
            continue
        if name not in targets_by_name:
            # Expected for most changes: the endpoint reports every changed
            # target, while the parse set only holds sidereal base targets of
            # READY/ONGOING observations in this run's programs. A renamed
            # target is re-created under its new name by the normal flow; the
            # old-name rows stay behind.
            _logger.info(
                f"Changed target {internal_id} ({name!r}) is not in this "
                f"run's parse set; skipping."
            )
            targets_not_in_parse_set += 1
            continue
        internal_ids_by_name.setdefault(name, []).append(internal_id)

    db_targets = await calc.target_repo.get_by_names(list(internal_ids_by_name))
    for name, internal_ids in internal_ids_by_name.items():
        db_target = db_targets.get(name)
        if db_target is None:
            # Relevant, but Sight has no row yet. The Stage-1 loop right after
            # creates it from these same fresh coordinates, so there is
            # nothing to update and nothing is lost.
            _logger.info(
                f"Changed target {name!r} is not in Sight yet; Stage 1 will "
                f"create it."
            )
            targets_pending_create += len(internal_ids)
            continue
        payload = targets_by_name[name]
        try:
            await calc.target_repo.update_fields(
                db_target,
                base_ra=payload.base_ra,
                base_dec=payload.base_dec,
                pm_ra=payload.pm_ra,
                pm_dec=payload.pm_dec,
                epoch=payload.epoch,
            )
        except Exception as exc:
            _logger.warning(f"Could not update changed target {name!r}: {exc}")
            targets_update_failed += len(internal_ids)
            continue
        targets_updated += len(internal_ids)
        labels_to_invalidate.update(labels_by_target.get(name, []))

    # Internal ids this run already parsed are mapped for free and are
    # schedulable by construction (the program dump uses the same workflow
    # filter). Only the leftovers need an ODB round trip, which drops any that
    # are not READY/ONGOING before their Stage 2 is touched.
    unmapped = [
        i for i in sorted(changes.observation_ids)
        if i not in labels_by_internal_id
    ]
    # A failed chunk is ignored here on purpose: one unresolved label means one
    # less invalidation this tick, and the next run retries the same window.
    resolved_labels = (
        (await resolve_schedulable_observation_labels(unmapped)).labels
        if unmapped else {}
    )
    for internal_id in sorted(changes.observation_ids):
        label = (
            labels_by_internal_id.get(internal_id)
            or resolved_labels.get(internal_id)
        )
        if label is None:
            # Not schedulable, deleted, or without a reference label.
            _logger.info(
                f"Changed observation {internal_id} is not schedulable or "
                f"could not be resolved; skipping."
            )
            obs_skipped += 1
            continue
        labels_to_invalidate.add(label)

    stage2_deleted = 0
    for label in sorted(labels_to_invalidate):
        stage2_deleted += await calc.visibility_repo.delete_by_observation(label)
    await calc.session.commit()

    targets_skipped = (
        targets_unresolved
        + targets_not_in_parse_set
        + targets_pending_create
        + targets_update_failed
    )
    counts = {
        "targets_updated": targets_updated,
        "targets_skipped": targets_skipped,
        "targets_not_in_parse_set": targets_not_in_parse_set,
        "targets_pending_create": targets_pending_create,
        "targets_unresolved": targets_unresolved,
        "targets_update_failed": targets_update_failed,
        "obs_invalidated": len(labels_to_invalidate),
        "obs_skipped": obs_skipped,
        "stage2_rows_deleted": stage2_deleted,
    }
    _logger.info(
        f"Applied ODB changes: {targets_updated} targets updated, "
        f"{targets_skipped} skipped ({targets_not_in_parse_set} not in parse "
        f"set, {targets_pending_create} awaiting Stage-1 create, "
        f"{targets_unresolved} unresolved, {targets_update_failed} failed); "
        f"Stage 2 invalidated for {len(labels_to_invalidate)} observations "
        f"({stage2_deleted} rows; {obs_skipped} not schedulable/unresolved)."
    )
    return counts


async def _store_missing_visibility(
    calc: Calculator,
    requests: list[ObservationRequest],
    start_date: date,
    end_date: date,
    heartbeat: Optional[Heartbeat],
    started_at: str,
) -> int:
    """Store Stage 2 only for (observation, night) pairs not already present.

    This is the "add only the ones not there" step: without it,
    ``store_visibility`` would recompute the whole semester on every cron tick.
    Commits per night to bound transaction size and persist progress.
    """
    if not requests:
        return 0

    obs_ids = [r.observation_id for r in requests]
    total_nights = (end_date - start_date).days + 1

    # Which (observation, night) pairs already exist, for the whole range in
    # one query. Checking night by night meant a query per night even when
    # every night was already covered, each one dragging back full rows
    # (visible_ranges and constraints JSONB) only to discard them.
    stored_pairs = await calc.visibility_repo.get_stored_observation_nights(
        obs_ids, start_date, end_date
    )

    loop_t0 = time.perf_counter()
    stored = 0
    current = start_date
    while current <= end_date:
        missing = [
            r for r in requests
            if (r.observation_id, current) not in stored_pairs
        ]
        nights_done = (current - start_date).days + 1
        loop_elapsed = time.perf_counter() - loop_t0

        if missing:
            t0 = time.perf_counter()
            result = await calc.store_visibility(missing, current, current)
            await calc.session.commit()
            elapsed = time.perf_counter() - t0
            night_stored = int(result.get("stored", 0))
            stored += night_stored

            loop_elapsed = time.perf_counter() - loop_t0
            # Moving ETA: measured seconds/night so far x nights remaining.
            eta = progress_eta_seconds(
                elapsed_seconds=loop_elapsed,
                done=nights_done,
                total=total_nights,
            )
            _logger.info(
                f"Stage 2 {current.isoformat()} [{nights_done}/{total_nights}]: "
                f"{len(missing)} missing obs, stored {night_stored} in {elapsed:.1f}s "
                f"({elapsed / len(missing) * 1000:.0f} ms/obs); "
                f"ETA ~{'unknown' if eta is None else _format_duration(eta)}."
            )

        if heartbeat is not None:
            await heartbeat(progress_detail(
                "stage2",
                done=nights_done,
                total=total_nights,
                unit="nights",
                elapsed_seconds=loop_elapsed,
                started_at=started_at,
                night=current.isoformat(),
                stored=stored,
            ))
        current += timedelta(days=1)

    return stored


async def run_aggregation(
    session: AsyncSession,
    *,
    heartbeat: Optional[Heartbeat] = None,
) -> dict:
    """Bring the Sight DB up to date for the current semester (sidereal only).

    session (AsyncSession): is the compute session (its own connection).
    heartbeat (Optional[Heartbeat]): is an optional async callback used to keep the coordination row fresh.
    """
    run_t0 = time.perf_counter()
    now = datetime.now(timezone.utc)
    started_at = now.isoformat()
    today = now.date()
    semester = Semester.find_semester_from_date(today)
    start_date = semester.start_date()
    end_date = semester.end_date()
    range_end = datetime.combine(end_date, datetime.min.time(), tzinfo=timezone.utc)

    # What changed in the ODB since the last successful run? Fetched before the
    # program dump; applied after parsing (the fresh payloads are needed).
    changes_t0 = time.perf_counter()
    changes, fetch_time = await _load_changes(session, now)
    changes_elapsed = time.perf_counter() - changes_t0
    if heartbeat is not None:
        await heartbeat({
            "phase": "changes",
            "started_at": started_at,
            "changed_observations": len(changes.observation_ids),
            "changed_targets": len(changes.target_ids),
        })

    labels = await gpp.client.scheduler.get_all_reference_labels()
    program_ids = [label[1] for label in labels]
    _logger.info(
        f"{len(program_ids)} available programs; semester {semester} "
        f"({start_date} .. {end_date})."
    )

    parse_t0 = time.perf_counter()
    targets_by_name, requests, labels_by_internal_id, counts = await _collect_requests(
        program_ids, range_end
    )
    parse_elapsed = time.perf_counter() - parse_t0
    _logger.info(
        f"Prepared {len(targets_by_name)} sidereal targets and {len(requests)} "
        f"observations in {parse_elapsed:.1f}s (skipped "
        f"{counts['skipped_nonsidereal']} non-sidereal, "
        f"{counts['skipped_no_target']} without a usable base target)."
    )
    if heartbeat is not None:
        await heartbeat({
            "phase": "parsed",
            "started_at": started_at,
            "programs": len(program_ids),
            "targets": len(targets_by_name),
            "observations": len(requests),
        })

    calc = Calculator(session)

    # Apply the ODB-reported changes before the Stage-1/Stage-2 flow below, so
    # updated targets are recomputed as stale and invalidated observations are
    # refilled as missing.
    apply_t0 = time.perf_counter()
    change_counts = await _apply_odb_changes(
        calc, changes, targets_by_name, requests, labels_by_internal_id
    )
    apply_elapsed = time.perf_counter() - apply_t0
    if heartbeat is not None:
        await heartbeat({
            "phase": "changes_applied", "started_at": started_at, **change_counts
        })

    targets = list(targets_by_name.values())
    batch_size = max(1, int(config.visibility_aggregator.target_batch_size))

    # Stage 1, in batches: create new targets (sight skips existing ones) and
    # ensure position arrays exist for every night in the range — including
    # gaps for targets that already existed (e.g. new semester nights).
    #
    # We commit per batch so a long semester-start backfill never holds one
    # giant transaction, and so progress persists.
    # A dyno killed mid-run just resumes from the remaining gaps on the next cron tick (every upsert is idempotent).
    created_total = 0
    stage1_t0 = time.perf_counter()
    for offset in range(0, len(targets), batch_size):
        chunk = targets[offset:offset + batch_size]
        batch_t0 = time.perf_counter()
        created = await calc.create_targets_bulk(chunk, start_date, end_date)
        created_total += created.created
        await calc.precompute_stage1(
            start_date, end_date, target_names=[t.name for t in chunk]
        )
        await session.commit()
        batch_elapsed = time.perf_counter() - batch_t0
        done = min(offset + batch_size, len(targets))
        loop_elapsed = time.perf_counter() - stage1_t0
        # Same moving ETA as Stage 2: measured seconds/target x targets left.
        eta = progress_eta_seconds(
            elapsed_seconds=loop_elapsed, done=done, total=len(targets)
        )
        _logger.info(
            f"Stage 1 batch {offset // batch_size + 1}: {len(chunk)} targets in "
            f"{batch_elapsed:.1f}s ({batch_elapsed / len(chunk):.2f}s/target); "
            f"{done}/{len(targets)} done; "
            f"ETA ~{'unknown' if eta is None else _format_duration(eta)}."
        )
        if heartbeat is not None:
            await heartbeat(progress_detail(
                "stage1",
                done=done,
                total=len(targets),
                unit="targets",
                elapsed_seconds=loop_elapsed,
                started_at=started_at,
                created=created_total,
            ))
    stage1_elapsed = time.perf_counter() - stage1_t0
    avg_target = stage1_elapsed / len(targets) if targets else 0.0
    _logger.info(
        f"Stage 1 done: {created_total} new targets; {len(targets)} targets "
        f"ensured over {start_date}..{end_date} in {stage1_elapsed:.1f}s "
        f"({avg_target:.2f}s/target avg)."
    )

    # Stage 2: store visibility for observations missing it on each night.
    stage2_t0 = time.perf_counter()
    stored = await _store_missing_visibility(
        calc, requests, start_date, end_date, heartbeat, started_at
    )
    stage2_elapsed = time.perf_counter() - stage2_t0
    _logger.info(f"Stage 2: stored {stored} new visibility rows in {stage2_elapsed:.1f}s.")

    # The whole run succeeded: advance the changes watermark to this run's
    # fetch time minus a safety overlap (clock skew; re-applying is idempotent).
    # A failed run never reaches this, so its window is retried next run.
    overlap = timedelta(
        seconds=float(config.visibility_aggregator.changes_overlap_seconds)
    )
    watermark_since = fetch_time - overlap
    await set_change_watermark(session, watermark_since)
    await session.commit()

    total_elapsed = time.perf_counter() - run_t0
    _logger.info(
        f"Aggregation timing: total={total_elapsed:.1f}s "
        f"(changes={changes_elapsed:.1f}s, apply={apply_elapsed:.1f}s, "
        f"parse={parse_elapsed:.1f}s, stage1={stage1_elapsed:.1f}s, "
        f"stage2={stage2_elapsed:.1f}s); {len(targets)} targets, "
        f"{stored} stage-2 rows inserted; ODB changes: "
        f"{len(changes.target_ids)} targets / {len(changes.observation_ids)} obs "
        f"reported, {change_counts['targets_updated']} targets updated "
        f"({change_counts['targets_skipped']} skipped: "
        f"{change_counts['targets_not_in_parse_set']} not in parse set, "
        f"{change_counts['targets_pending_create']} awaiting Stage-1 create, "
        f"{change_counts['targets_unresolved']} unresolved, "
        f"{change_counts['targets_update_failed']} failed), "
        f"{change_counts['obs_invalidated']} obs invalidated "
        f"({change_counts['stage2_rows_deleted']} stage-2 rows deleted, "
        f"{change_counts['obs_skipped']} not schedulable/unresolved)."
    )

    return {
        "semester": str(semester),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "programs": len(program_ids),
        "targets": len(targets),
        "targets_created": created_total,
        "observations": len(requests),
        "stage2_stored": stored,
        "elapsed_seconds": round(total_elapsed, 1),
        "changes_seconds": round(changes_elapsed, 1),
        "apply_seconds": round(apply_elapsed, 1),
        "parse_seconds": round(parse_elapsed, 1),
        "stage1_seconds": round(stage1_elapsed, 1),
        "stage2_seconds": round(stage2_elapsed, 1),
        "changed_observations": len(changes.observation_ids),
        "changed_targets": len(changes.target_ids),
        "watermark_since": watermark_since.isoformat(),
        **change_counts,
        **counts,
    }
