#!/usr/bin/env python3
"""Which programs and observations the aggregator's parse loses, and why.

Read-only, ODB only. It never opens the Sight DB and never touches the
coordination row, so it is safe to run while a night is in progress and the
aggregator itself is held off. Nothing imports it.

It replays ``aggregator._collect_requests``, except that each observation is
parsed inside its own try block. The aggregator wraps a whole program in one, so
the first observation that raises takes every remaining observation of that
program with it and the log says only "Failed to process a program". Isolating
them names the observation that raised and shows how many others its program
lost with it.

Usage (from the repo root, with GPP credentials in the environment):

    .venv/bin/python scripts/check_parse_errors.py
    .venv/bin/python scripts/check_parse_errors.py --programs G-2026A-0001
    .venv/bin/python scripts/check_parse_errors.py --traceback
"""

import argparse
import asyncio
import logging
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from lucupy.types import ZeroTime  # noqa: E402

from scheduler.clients.gpp import gpp  # noqa: E402
from scheduler.core.programprovider.gpp import (  # noqa: E402
    GppProgramProvider,
    gpp_program_data,
)
from scheduler.core.programprovider.gpp.gppprogramprovider import (  # noqa: E402
    get_progid,
)
from scheduler.core.sources.sources import Sources  # noqa: E402
from scheduler.services.sight._temporary.lucupy_adapters import (  # noqa: E402
    stage2_constraints,
    target_create,
)

# Private, but imported rather than redefined so it cannot drift from the
# aggregator this script mirrors.
from scheduler.services.visibility_aggregator.aggregator import (  # noqa: E402
    _OBS_CLASSES,
)


def _program_label(data: dict) -> str:
    """The program's reference label, however the payload carries it."""
    try:
        reference = data.get("reference")
        if reference and reference.get("label"):
            return str(reference["label"])
        return str(get_progid(data["root"]))
    except Exception:
        return f"<unlabelled {data.get('id', '?')}>"


def _describe(exc: BaseException) -> str:
    """The exception, plus the provider line that raised it."""
    frames = traceback.extract_tb(exc.__traceback__)
    where = ""
    if frames:
        # The deepest frame inside the scheduler, not inside a library.
        ours = [f for f in frames if "scheduler/" in f.filename] or frames
        last = ours[-1]
        where = f" [{Path(last.filename).name}:{last.lineno} in {last.name}]"
    return f"{type(exc).__name__}: {exc}{where}"


class _ProviderWarnings(logging.Handler):
    """Collect what the provider logs while it parses.

    ``parse_program`` returns None from several places and says which one only
    in a warning — an empty root group, an undeterminable semester, an unknown
    program type. Capturing those turns "returned None" into the actual reason
    without reaching into the provider.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if "programprovider" in record.name:
            self.messages.append(record.getMessage())

    def drain(self) -> str:
        messages, self.messages = self.messages, []
        return " | ".join(messages)


async def _check(program_ids: Optional[list], show_traceback: bool) -> int:
    provider = GppProgramProvider(_OBS_CLASSES, Sources())
    warnings = _ProviderWarnings()
    logging.getLogger().addHandler(warnings)

    program_problems: list[tuple[str, str, str]] = []  # label, reason, detail
    noisy_programs: list[tuple[str, str]] = []  # label, warnings
    obs_problems: list[tuple[str, str, str, str]] = []  # program, obs, reason, detail
    reasons = Counter()
    programs_seen = 0
    observations_seen = 0
    observations_ok = 0

    async for raw in await gpp_program_data(program_ids):
        programs_seen += 1
        data = next(iter(raw.values())) if len(raw.keys()) == 1 else raw
        label = _program_label(data)

        # --- program level ---------------------------------------------------
        warnings.drain()
        try:
            program = provider.parse_program(data)
        except Exception as exc:
            # In the aggregator this is the `bad_programs` counter: every
            # observation of this program is lost, and none of them is named.
            detail = _describe(exc)
            logged = warnings.drain()
            program_problems.append(
                (
                    label,
                    "PROGRAM_PARSE_ERROR",
                    f"{detail}{f' | {logged}' if logged else ''}",
                )
            )
            reasons["PROGRAM_PARSE_ERROR"] += 1
            if show_traceback:
                traceback.print_exc()
            continue

        if program is None:
            # parse_program returns None from several places and names the one
            # it took only in a warning, so that is the detail worth showing.
            logged = warnings.drain()
            program_problems.append(
                (
                    label,
                    "PROGRAM_NOT_PARSED",
                    logged or "parse_program returned None (nothing logged)",
                )
            )
            reasons["PROGRAM_NOT_PARSED"] += 1
            continue

        # Parsed, but the provider still complained: usually an observation or
        # group it dropped on its own, which is why the program is short.
        logged = warnings.drain()
        if logged:
            noisy_programs.append((label, logged))

        if program.program_awarded() == ZeroTime:
            program_problems.append(
                (label, "NO_AWARDED_TIME", "program_awarded() == 0")
            )
            reasons["NO_AWARDED_TIME"] += 1
            continue

        # --- observation level ------------------------------------------------
        try:
            observations = list(program.observations())
        except Exception as exc:
            program_problems.append((label, "OBSERVATIONS_UNREADABLE", _describe(exc)))
            reasons["OBSERVATIONS_UNREADABLE"] += 1
            if show_traceback:
                traceback.print_exc()
            continue

        for obs in observations:
            observations_seen += 1
            obs_id = getattr(getattr(obs, "id", None), "id", "<no id>")
            try:
                base = obs.base_target()
                if base is None:
                    obs_problems.append((label, obs_id, "NO_BASE_TARGET", ""))
                    reasons["NO_BASE_TARGET"] += 1
                    continue

                payload = target_create(base)
                if payload is None:
                    obs_problems.append(
                        (
                            label,
                            obs_id,
                            "UNSUPPORTED_TARGET",
                            f"target_create returned None for {type(base).__name__}",
                        )
                    )
                    reasons["UNSUPPORTED_TARGET"] += 1
                    continue

                if not payload.is_sidereal:
                    obs_problems.append(
                        (label, obs_id, "NON_SIDEREAL", str(payload.name))
                    )
                    reasons["NON_SIDEREAL"] += 1
                    continue

                if obs.site is None:
                    obs_problems.append((label, obs_id, "NO_SITE", ""))
                    reasons["NO_SITE"] += 1
                    continue

                # The last thing _collect_requests does per observation, and the
                # most likely place for a surprise: it walks constraints and
                # expands timing windows.
                stage2_constraints(
                    obs,
                    has_resources=True,
                    can_schedule=True,
                    range_end=program.end,
                    program_start=program.start,
                    program_end=program.end,
                )
            except Exception as exc:
                # This is the one the aggregator never names: in it, this
                # exception aborts the whole program.
                obs_problems.append(
                    (label, obs_id, "OBSERVATION_ERROR", _describe(exc))
                )
                reasons["OBSERVATION_ERROR"] += 1
                if show_traceback:
                    traceback.print_exc()
                continue

            observations_ok += 1

    logging.getLogger().removeHandler(warnings)
    _report(
        programs_seen,
        observations_seen,
        observations_ok,
        program_problems,
        noisy_programs,
        obs_problems,
        reasons,
    )
    return 1 if reasons else 0


def _report(
    programs_seen,
    observations_seen,
    observations_ok,
    program_problems,
    noisy_programs,
    obs_problems,
    reasons,
) -> None:
    print()
    print(f"Programs read:      {programs_seen}")
    print(f"Observations read:  {observations_seen}")
    print(f"Would be requested: {observations_ok}")
    print()

    if program_problems:
        print("Programs lost (every observation in them goes with it):")
        for label, reason, detail in sorted(program_problems):
            print(f"  {label:<24} {reason:<20} {detail}")
        print()

    if noisy_programs:
        print(
            "Programs that parsed but logged warnings (usually a dropped "
            "observation or group):"
        )
        for label, logged in sorted(noisy_programs):
            print(f"  {label:<24} {logged}")
        print()

    if obs_problems:
        # Grouped by program: a program with many failures is usually one cause.
        print("Observations lost:")
        by_program: dict[str, list] = {}
        for label, obs_id, reason, detail in obs_problems:
            by_program.setdefault(label, []).append((obs_id, reason, detail))
        for label in sorted(by_program):
            rows = by_program[label]
            errors = sum(1 for _, reason, _ in rows if reason == "OBSERVATION_ERROR")
            note = (
                (
                    f"  <- in the aggregator the first of these {errors} error(s) "
                    f"aborts the whole program"
                )
                if errors
                else ""
            )
            print(f"  {label} ({len(rows)}){note}")
            for obs_id, reason, detail in sorted(rows):
                print(f"      {obs_id:<26} {reason:<20} {detail}")
        print()

    if reasons:
        print("Totals by reason:")
        for reason, count in reasons.most_common():
            print(f"  {reason:<24} {count}")
    else:
        print("Nothing failed to parse.")


async def _main(args) -> int:
    # The ODB dump is keyed by internal program id, so reference labels typed on
    # the command line are translated here rather than silently matching nothing.
    labels = await gpp.client.scheduler.get_all_reference_labels()
    ids_by_label = {str(label): str(program_id) for label, program_id in labels}

    if args.programs:
        programs = [
            ids_by_label.get(entry.strip(), entry.strip())
            for entry in args.programs.split(",")
        ]
        unknown = [
            e
            for e in args.programs.split(",")
            if e.strip().startswith("G-") and e.strip() not in ids_by_label
        ]
        if unknown:
            print(
                f"Not in the available-programs list, passed through as-is: {unknown}"
            )
    else:
        programs = list(ids_by_label.values())
        print(f"{len(programs)} available programs.")
    try:
        return await _check(programs, args.traceback)
    finally:
        # The client holds sockets bound to this loop; close them before it ends.
        await gpp.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--programs",
        help="Comma-separated programs to check, as reference labels (G-2026A-0001) "
        "or ODB ids. Defaults to every available program.",
    )
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="Print the full traceback for each failure, not just its message.",
    )
    sys.exit(asyncio.run(_main(parser.parse_args())))
