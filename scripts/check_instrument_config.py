#!/usr/bin/env python3
"""Does each observation's FPU, disperser and filter survive parsing into the Plan table?

Read-only, ODB only. For every matching observation it prints three layers side
by side, so a value that goes missing shows exactly where it was lost:

    raw ODB     what GPP returned in observingMode (the populated mode block)
    parsed      the resources GppProgramProvider.parse_observing_mode built
    plan table  what SVisit.from_computed_visit would send to the UI, read
                through the same Observation.fpu()/disperser()/filters() calls

An observation is flagged when the ODB has a value and the Plan table does not.

Usage (from the repo root, with GPP credentials in the environment):

    .venv/bin/python scripts/check_instrument_config.py --programs G-2026B-0123
    .venv/bin/python scripts/check_instrument_config.py --obs G-2026B-0123-0004 --dump
    .venv/bin/python scripts/check_instrument_config.py --instrument GNIRS
    .venv/bin/python scripts/check_instrument_config.py --instrument all --programs G-2026B-0123
"""

import argparse
import asyncio
import json
import sys
import traceback
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from lucupy.minimodel import ResourceType  # noqa: E402

from scheduler.clients.gpp import gpp  # noqa: E402
from scheduler.core.programprovider.gpp import (  # noqa: E402
    GppProgramProvider,
    gpp_program_data,
)
from scheduler.core.sources.sources import Sources  # noqa: E402
from scheduler.services.visibility_aggregator.aggregator import (  # noqa: E402
    _OBS_CLASSES,
)

# Keys inside the mode block that carry a value the Plan table should show.
_FPU_KEYS = {"fpu", "custom_mask"}
_DISPERSER_KEYS = {"grating", "disperser"}
_FILTER_KEYS = {"filter_", "filters"}
_OTHER_KEYS = {"prism", "decker", "central_wavelength", "central_wavelengths"}
_RAW_KEYS = _FPU_KEYS | _DISPERSER_KEYS | _FILTER_KEYS | _OTHER_KEYS

# initial_/default_/explicit_ are the inputs GPP resolves into the plain key, and
# the acquisition and telescope-config branches describe other steps, so none of
# them is what the science sequence uses.
_SKIP_PREFIXES = ("initial_", "default_", "explicit_")
_SKIP_BRANCHES = {"acquisition", "telescope_configs"}


def _plain(value: Any) -> Any:
    """The payload with enums replaced by their values, so it prints and dumps cleanly."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def _mode_block(observing_mode: dict) -> tuple[Optional[str], Optional[dict]]:
    """The one populated mode block (gnirs_spectroscopy, gmos_north_long_slit, ...)."""
    for key, value in observing_mode.items():
        if key not in ("instrument", "mode") and isinstance(value, dict) and value:
            return key, value
    return None, None


def _raw_values(block: dict, path: str = "") -> Iterator[tuple[str, Any]]:
    """(dotted path, value) for every interesting key, wherever it is nested.

    Nesting matters: GNIRS keeps its FPU at slit.fpu or ifu.fpu, not at the top
    of the block the way GMOS and F2 do.
    """
    for key, value in block.items():
        if key in _SKIP_BRANCHES or key.startswith(_SKIP_PREFIXES) or value is None:
            continue
        here = f"{path}.{key}" if path else key
        if key in _RAW_KEYS:
            yield here, _plain(value)
        elif isinstance(value, dict):
            yield from _raw_values(value, here)


def _observations(node: Any) -> Iterator[dict]:
    """Every raw observation dict in a program tree, however deep the groups go."""
    if isinstance(node, dict):
        if "observing_mode" in node:
            yield node
            return
        for value in node.values():
            yield from _observations(value)
    elif isinstance(node, list):
        for value in node:
            yield from _observations(value)


def _obs_label(data: dict) -> str:
    reference = data.get("reference")
    if reference and reference.get("label"):
        return str(reference["label"])
    return str(data.get("id", "<no id>"))


def _resources_by_type(resources) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for r in sorted(resources, key=lambda r: (r.type.name, r.id)):
        name = (
            r.id
            if not r.description or r.description == r.id
            else f"{r.id}({r.description})"
        )
        grouped.setdefault(r.type.name, []).append(name)
    return grouped


def _plan_row(obs) -> dict[str, Any]:
    """Exactly the strings SVisit.from_computed_visit puts in the Plan table."""
    instrument = obs.instrument()
    fpu = obs.fpu()
    disperser = obs.disperser()
    filters = obs.filters()
    return {
        "instrument": instrument.id if instrument is not None else "None",
        "fpu": fpu.id if fpu is not None else "None",
        "disperser": disperser.id if disperser is not None else "None",
        "filters": [f.id for f in filters] if filters is not None else [],
    }


def _missing(raw: list[tuple[str, Any]], plan: Optional[dict]) -> list[str]:
    """Fields the ODB filled and the Plan table shows empty."""
    if plan is None:
        return []

    def raw_for(keys: set) -> Optional[str]:
        hits = [f"{p}={v}" for p, v in raw if p.split(".")[-1] in keys and v]
        return ", ".join(hits) if hits else None

    problems = []
    for field, keys, empty in (
        ("fpu", _FPU_KEYS, plan["fpu"] == "None"),
        ("disperser", _DISPERSER_KEYS, plan["disperser"] == "None"),
        ("filters", _FILTER_KEYS, not plan["filters"]),
    ):
        source = raw_for(keys)
        if source and empty:
            problems.append(f"{field} (raw {source})")
    return problems


async def _check(
    program_ids: list, instrument: str, obs_filter: Optional[set], dump: bool
) -> int:
    provider = GppProgramProvider(_OBS_CLASSES, Sources())
    totals = Counter()

    async for raw in await gpp_program_data(program_ids):
        data = next(iter(raw.values())) if len(raw.keys()) == 1 else raw

        # The full parse gives the Observation objects the Plan table reads from.
        parsed_by_id: dict[str, Any] = {}
        program_error = None
        try:
            program = provider.parse_program(data)
            if program is not None:
                parsed_by_id = {o.id.id: o for o in program.observations()}
            else:
                program_error = "parse_program returned None"
        except Exception as exc:
            program_error = f"{type(exc).__name__}: {exc}"

        for obs_data in _observations(data.get("root", data)):
            label = _obs_label(obs_data)
            if obs_filter and label not in obs_filter:
                continue
            observing_mode = obs_data.get("observing_mode") or {}
            inst = str(_plain(observing_mode.get("instrument")))
            ocs_inst = GppProgramProvider._gpp_inst_to_ocs.get(inst, inst)
            if instrument != "ALL" and instrument not in (
                inst.upper(),
                ocs_inst.upper(),
            ):
                continue

            totals["checked"] += 1
            mode = _plain(observing_mode.get("mode"))
            block_key, block = _mode_block(observing_mode)
            raw_vals = list(_raw_values(block)) if block else []

            print(f"\n{label}  {ocs_inst}  {mode}")
            print(
                f"  raw ODB    [{block_key}]: "
                + ("  ".join(f"{p}={v}" for p, v in raw_vals) or "(no mode block)")
            )
            if dump and block:
                print(
                    "  "
                    + json.dumps(_plain(block), indent=2, default=str).replace(
                        "\n", "\n  "
                    )
                )

            try:
                resources, wavelength, _ = provider.parse_observing_mode(observing_mode)
                grouped = _resources_by_type(resources)
                print(
                    f"  parsed     : "
                    + "  ".join(f"{t}={v}" for t, v in grouped.items())
                    + f"  wavelength={wavelength}"
                )
                if ResourceType.FPU.name not in grouped:
                    totals["no FPU resource from parse_observing_mode"] += 1
            except Exception as exc:
                totals["parse_observing_mode raised"] += 1
                print(f"  parsed     : RAISED {type(exc).__name__}: {exc}")
                if dump:
                    traceback.print_exc()

            obs = parsed_by_id.get(label)
            plan = None
            if obs is not None:
                plan = _plan_row(obs)
                print(
                    f"  plan table : " + "  ".join(f"{k}={v}" for k, v in plan.items())
                )
            else:
                reason = (
                    program_error
                    or "not in the parsed program (inactive, wrong state or class?)"
                )
                print(f"  plan table : n/a, {reason}")
                totals["not in parsed program"] += 1

            problems = _missing(raw_vals, plan)
            if problems:
                print(f"  -> MISSING in Plan table: {'; '.join(problems)}")
                totals["flagged"] += 1
                for p in problems:
                    totals[f"missing {p.split(' ')[0]}"] += 1
            elif plan is not None:
                print("  -> OK")

    print("\nSummary:")
    if not totals:
        print("  No matching observations.")
    for key, count in totals.most_common():
        print(f"  {key:<45} {count}")
    return 1 if totals["flagged"] else 0


async def _main(args) -> int:
    obs_filter = {o.strip() for o in args.obs.split(",")} if args.obs else None
    wanted = [p.strip() for p in args.programs.split(",")] if args.programs else []
    # An observation label is its program label plus "-NNNN", so --obs alone is enough.
    if obs_filter and not wanted:
        wanted = sorted({o.rsplit("-", 1)[0] for o in obs_filter})

    # The ODB dump is keyed by internal program id, so translate reference labels.
    labels = await gpp.client.scheduler.get_all_reference_labels()
    ids_by_label = {str(label): str(program_id) for label, program_id in labels}
    if wanted:
        unknown = [p for p in wanted if p.startswith("G-") and p not in ids_by_label]
        if unknown:
            print(
                f"Not in the available-programs list, passed through as-is: {unknown}"
            )
        programs = [ids_by_label.get(p, p) for p in wanted]
    else:
        programs = list(ids_by_label.values())
        print(f"{len(programs)} available programs.")

    try:
        return await _check(programs, args.instrument.upper(), obs_filter, args.dump)
    finally:
        # The client holds sockets bound to this loop; close them before it ends.
        await gpp.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--programs",
        help="Comma-separated program reference labels or ODB ids. "
        "Defaults to every available program.",
    )
    parser.add_argument(
        "--obs",
        help="Comma-separated observation reference labels, e.g. G-2026B-0123-0004.",
    )
    parser.add_argument(
        "--instrument",
        default="GNIRS",
        help="GPP or OCS instrument name to check (GNIRS, GMOS_NORTH, GMOS-N, ...), "
        "or 'all'. Default: GNIRS.",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Also print the full raw mode block as JSON.",
    )
    sys.exit(asyncio.run(_main(parser.parse_args())))
