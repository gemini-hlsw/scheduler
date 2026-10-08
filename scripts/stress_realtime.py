#!/usr/bin/env python3
"""Realtime stress test: how long a plan takes with up to 1500 observations per site.

Drives ``EngineRT.compute_event_plan`` directly, the same call the operation process
makes for every event. Every step is measured with the OTel ``timed()`` spans the
scheduler already has. Steps that have no span yet are wrapped from here (named
``stress.*``), so the production code is not touched.

Two sources, both live (GPP credentials in the environment for every command):

- clones (default): the programs active today are fetched from the ODB once, at the
  start of the command, through the same call realtime makes, then cloned until each
  site holds N observations. The fetch is outside the plan numbers.
- odb: no clones; every plan fetches from the ODB itself, as realtime does, and the
  fetch is part of the plan numbers.

Usage (from the repo root), in this order:

    # 1. Smoke test, clones, local visibility.
    .venv/bin/python scripts/stress_realtime.py run --sizes 25 --plans 1 \\
        --visibility local --label smoke

    # 2. Full local sweep (the engine reads the whole semester; long).
    .venv/bin/python scripts/stress_realtime.py run --sizes 250,500,1000,1500 --plans 3 \\
        --visibility local --label local

    # 3. Sight with a full-semester DB (docker compose up -d postgres; alembic upgrade head).
    DATABASE_URL=postgresql+asyncpg://scheduler:scheduler_dev@localhost:5433/scheduler \\
        .venv/bin/python scripts/stress_realtime.py seed-sight --window full
    DATABASE_URL=postgresql+asyncpg://scheduler:scheduler_dev@localhost:5433/scheduler \\
        .venv/bin/python scripts/stress_realtime.py run --sizes 250,500,1000,1500 --plans 3 \\
        --visibility sight --label sight

    # 4. Live ODB data only, fetched inside every plan.
    .venv/bin/python scripts/stress_realtime.py run --source odb --plans 3 \\
        --visibility local --label live-local
    DATABASE_URL=postgresql+asyncpg://scheduler:scheduler_dev@localhost:5433/scheduler \\
        .venv/bin/python scripts/stress_realtime.py run --source odb --plans 3 \\
        --visibility sight --label live-sight

Every run reads visibility to the end of the night's semester (BuildParameters.
visibility_end), because remaining visibility, and so every score, is summed over the
nights the Collector reads. --vis-end realtime-default reads only the 15 nights
default_operation_parameters gives; that is also the only window a --window narrow
seed can serve.

Results land in stress/results/report.md, one section per --label, with a comparison
table across labels on top. Add OTEL_EXPORTER_OTLP_ENDPOINT (and its headers) to any
``run`` to also push the metrics to Grafana, tagged deployment.environment=stress.

Clones are rebuilt from the live ODB every time, so run a Sight test soon after its
seed-sight, with the same --night, --cap and --programs. If the ODB changed in between,
the clones change too, and the run refuses to start until Sight is seeded again.
"""

import argparse
import asyncio
import copy
import functools
import inspect
import json
import logging
import os
import random
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))

STRESS_DIR = REPO / "stress"
REPORT_PATH = STRESS_DIR / "results" / "report.md"

PLAN_TARGET_S = 60.0
NIGHTS_IN_WINDOW = 15            # what default_operation_parameters gives a realtime build
CLONE_NUMBER_BASE = 9000         # clone program numbers start here, clear of real programs
DEFAULT_SIZES = "250,500,1000,1500"
DEFAULT_CAP = 1500
DEFAULT_SEED = 1041
LIVE = 0                         # size key for --source odb: whatever the ODB holds

# GPP internal ids: one letter, a dash, hex (p-1a2b, o-7f, g-3c, t-9e).
_INTERNAL_ID = re.compile(r"^[a-z]-[0-9a-f]+$")
# Science program reference labels, with or without a type suffix: G-2026B-0331-Q.
_PROGRAM_LABEL = re.compile(r"^(G-\d{4}[AB])-(\d+)(-[A-Z]+)?$")
_LABEL_ARG = re.compile(r"^[A-Za-z0-9_.-]+$")


# --------------------------------------------------------------------------------------
# Environment and imports
# --------------------------------------------------------------------------------------

def _prepare_env(visibility: str) -> None:
    """Pin the mode before anything under scheduler.* is imported.

    The mode and the visibility strategy are both read once at import time
    (core/builder/modes.py, config.yaml), so setting them later does nothing.
    """
    os.environ["SCHEDULER_MODE"] = "REALTIME"
    os.environ["COLLECTOR_VISIBILITY_STRATEGY"] = visibility
    # The interlock would make every plan wait on the coordination row.
    os.environ["VIS_AGG_INTERLOCK"] = "false"


def _quiet_scheduler_logs(level: str) -> None:
    """Every scheduler module logger carries its own level and handler."""
    numeric = getattr(logging, level)
    for name in list(logging.Logger.manager.loggerDict):
        if name.startswith("scheduler") and name != "scheduler.perf":
            logging.getLogger(name).setLevel(numeric)


def _require_database_url(command: str) -> None:
    if not os.environ.get("DATABASE_URL"):
        sys.exit(f"{command} needs DATABASE_URL pointing at the local docker-compose Postgres, e.g.\n"
                 "  DATABASE_URL=postgresql+asyncpg://scheduler:scheduler_dev@localhost:5433/scheduler")


def _night_anchor(night: Optional[str]) -> datetime:
    """08:00 UT of the date whose morning ends the night, as realtime builds use it."""
    from scheduler.engine.params import default_operation_start
    if night is None:
        return default_operation_start()
    try:
        day = date.fromisoformat(night)
    except ValueError:
        sys.exit(f"--night must be YYYY-MM-DD, got {night!r}")
    return datetime(day.year, day.month, day.day, 8, tzinfo=UTC)


def _vis_end(night_start: datetime, value: Optional[str]) -> datetime:
    """Last night the engine reads, as BuildParameters.visibility_end (inclusive).

    None or 'realtime-default' gives the 15 nights of default_operation_parameters.
    'semester' mirrors SchedulerParameters(semester_visibility=True): the semester's last
    local date, plus one day for UT, at the anchor's hour.
    """
    if value is None or value == "realtime-default":
        return night_start + timedelta(days=NIGHTS_IN_WINDOW - 1)
    if value == "semester":
        from lucupy.minimodel.semester import Semester
        end = Semester.find_semester_from_date(night_start - timedelta(days=1)).end_date() + timedelta(days=1)
        return datetime(end.year, end.month, end.day, night_start.hour, tzinfo=UTC)
    try:
        day = date.fromisoformat(value)
    except ValueError:
        sys.exit(f"--vis-end must be 'semester', 'realtime-default' or YYYY-MM-DD, got {value!r}")
    end = datetime(day.year, day.month, day.day, night_start.hour, tzinfo=UTC)
    if end < night_start:
        sys.exit(f"--vis-end {day} is before the night ({night_start.date()}).")
    return end


def _window_nights(night_start: datetime, vis_end: datetime) -> List[date]:
    """The UT dates the Collector's time grid holds, one per night."""
    return [night_start.date() + timedelta(days=n)
            for n in range((vis_end.date() - night_start.date()).days + 1)]


def _check_ops_calendar(night_start: datetime, vis_end: Optional[datetime] = None) -> None:
    """OpsResourceService raises for dates outside its calendar; fail before any work."""
    from lucupy.minimodel import ALL_SITES
    from scheduler.core.sources import Origins
    from scheduler.core.sources.sources import Sources

    sources = Sources()
    sources.set_origin(Origins.OPS())
    resource = sources.origin.resource
    nights = _window_nights(night_start, vis_end or _vis_end(night_start, None))
    # Collector.night_configurations asks for each night's UT date minus one day.
    first = nights[0] - timedelta(days=1)
    last = nights[-1] - timedelta(days=1)
    for site in ALL_SITES:
        earliest, latest = resource.date_range_for_site(site)
        if first < earliest or last > latest:
            sys.exit(f"Night {night_start.date()} needs resource data for {site.name} from {first} to "
                     f"{last}, but the ops calendar covers {earliest}..{latest} "
                     f"(services/resource/data/operation/telescope_schedules.xlsx, and "
                     f"pickles/opsresource.pickle may be stale). Pick another --night or --vis-end.")


# --------------------------------------------------------------------------------------
# Live source programs
# --------------------------------------------------------------------------------------

def _program_body(raw: dict) -> dict:
    """The program dict, however get_all wrapped it."""
    return next(iter(raw.values())) if len(raw.keys()) == 1 else raw


def _iter_raw_observations(group: Optional[dict]) -> Iterable[dict]:
    for element in (group or {}).get("elements") or []:
        if element.get("observation"):
            yield element["observation"]
        elif element.get("group"):
            yield from _iter_raw_observations(element["group"])


def _raw_site_counts(programs: List[dict]) -> Counter:
    """Observations per site straight from the payload, before any parsing."""
    from scheduler.core.programprovider.gpp import GppProgramProvider
    counts: Counter = Counter()
    for raw in programs:
        for obs in _iter_raw_observations(_program_body(raw).get("root")):
            instrument = obs.get("instrument")
            site = GppProgramProvider._site_for_inst.get(getattr(instrument, "value", instrument))
            counts[site.name if site else "unknown"] += 1
    return counts


def _short(exc: BaseException, limit: int = 160) -> str:
    text = " ".join(str(exc).split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


async def _resolve_program_ids(text: str) -> List[str]:
    """Reference labels (G-2026B-0001) or ODB ids, as gpp_program_data takes them: ODB ids."""
    from scheduler.clients.gpp import gpp
    labels = await gpp.client.scheduler.get_all_reference_labels()
    ids_by_label = {str(label): str(program_id) for label, program_id in labels}
    wanted = [p.strip() for p in text.split(",") if p.strip()]
    unknown = [p for p in wanted if p.startswith("G-") and p not in ids_by_label]
    if unknown:
        sys.exit(f"Not in the ODB's available-programs list: {unknown}")
    return [ids_by_label.get(p, p) for p in wanted]


async def _fetch_sources(programs: Optional[str]) -> dict:
    """The programs to clone, fetched now through the call realtime makes.

    gpp_program_data is the realtime loading path, including the provider's own
    exclude_programs list. With no --programs it brings every program active today.
    """
    from scheduler.core.programprovider.gpp import gpp_program_data

    program_ids = await _resolve_program_ids(programs) if programs else None
    print("Fetching source programs from the ODB...")
    started = time.perf_counter()
    data = await gpp_program_data(program_ids)
    fetched = [item async for item in data]
    seconds = time.perf_counter() - started
    counts = _raw_site_counts(fetched)
    meta = {
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "programs": len(fetched),
        "observations": sum(counts.values()),
        "observations_per_site": dict(counts),
        "fetch_seconds": seconds,
        "program_filter": programs,
    }
    print(f"Fetched {len(fetched)} programs, {meta['observations']} observations "
          f"({', '.join(f'{k} {v}' for k, v in sorted(counts.items()))}) in {seconds:.1f}s.")
    if not fetched:
        sys.exit("The ODB returned no programs.")
    return {"meta": meta, "programs": fetched}


# --------------------------------------------------------------------------------------
# Clone generator
# --------------------------------------------------------------------------------------

def _parse_sex(text: str) -> float:
    """'HH:MM:SS.s' or '[+-]DD:MM:SS.s' to decimal, keeping the sign of '-00:..'."""
    t = text.strip()
    negative = t.startswith("-")
    a, b, c = (float(x) for x in t.lstrip("+-").split(":"))
    value = a + b / 60.0 + c / 3600.0
    return -value if negative else value


def _fmt_sex(value: float, signed: bool) -> str:
    negative = value < 0
    micro_total = round(abs(value) * 3600 * 1_000_000)
    whole, micro = divmod(micro_total, 1_000_000)
    hh, rem = divmod(whole, 3600)
    mm, ss = divmod(rem, 60)
    if not signed:
        hh %= 24
    text = f"{hh:02d}:{mm:02d}:{ss:02d}.{micro:06d}"
    if signed:
        return ("-" if negative else "+") + text
    return text


def _like(original: Any, value: float) -> Any:
    """`value` in the numeric type the payload already used for that field."""
    if isinstance(original, bool):
        return original
    if isinstance(original, int):
        return int(round(value))
    if isinstance(original, Decimal):
        return Decimal(repr(round(value, 9)))
    if isinstance(original, str):
        return repr(round(value, 9))
    return float(value)


def _set_ra(ra: dict, degrees: float) -> None:
    hours = degrees / 15.0
    for key, value in list(ra.items()):
        if key == "hms":
            ra[key] = _fmt_sex(hours, signed=False)
        elif key == "hours":
            ra[key] = _like(value, hours)
        elif key == "degrees":
            ra[key] = _like(value, degrees)
        elif key == "microarcseconds":
            ra[key] = _like(value, degrees * 3600e6)
        elif key == "microseconds":
            ra[key] = _like(value, hours * 3600e6)


def _set_dec(dec: dict, degrees: float) -> None:
    for key, value in list(dec.items()):
        if key == "dms":
            dec[key] = _fmt_sex(degrees, signed=True)
        elif key == "degrees":
            # Some payloads carry Dec on 0..360 (see parse_sidereal_target); keep theirs.
            wrapped = isinstance(value, (int, float, Decimal)) and float(value) > 90
            dec[key] = _like(value, degrees % 360 if wrapped else degrees)
        elif key == "microarcseconds":
            wrapped = isinstance(value, (int, float, Decimal)) and float(value) > 90 * 3600e6
            dec[key] = _like(value, (degrees % 360 if wrapped else degrees) * 3600e6)


def _jitter_target(target: dict, rng: random.Random, ra_spread: float, dec_spread: float) -> None:
    sidereal = target["sidereal"]
    ra = sidereal.get("ra") or {}
    dec = sidereal.get("dec") or {}
    if "hms" not in ra or "dms" not in dec:
        return
    new_ra = (_parse_sex(ra["hms"]) * 15.0 + rng.uniform(-ra_spread, ra_spread)) % 360.0
    new_dec = max(-89.5, min(89.5, _parse_sex(dec["dms"]) + rng.uniform(-dec_spread, dec_spread)))
    # lucupy's sex2dec reads '-00:..' as positive, so never write a Dec in (-1, 0).
    if -1.0 < new_dec < 0.0:
        new_dec = -1.0
    _set_ra(ra, new_ra)
    _set_dec(dec, new_dec)


def _collect_internal_ids(node: Any, out: set) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "id" and isinstance(value, str) and _INTERNAL_ID.match(value):
                out.add(value)
            elif isinstance(value, (dict, list)):
                _collect_internal_ids(value, out)
    elif isinstance(node, list):
        for value in node:
            _collect_internal_ids(value, out)


@dataclass
class _CloneContext:
    id_map: Dict[str, str]
    old_label: str
    new_label: str
    name_suffix: str
    seed: str
    ra_spread: float
    dec_spread: float
    jittered: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)

    def map_str(self, value: str) -> str:
        mapped = self.id_map.get(value)
        if mapped is not None:
            return mapped
        if value == self.old_label:
            return self.new_label
        if value.startswith(self.old_label + "-"):
            return self.new_label + value[len(self.old_label):]
        return value

    def target(self, node: dict) -> None:
        """Unique name and moved coordinates, the same for every copy of one target."""
        name = node["name"]
        key = str(node.get("id") or name)
        sidereal = node["sidereal"]
        if key not in self.jittered:
            rng = random.Random(f"{self.seed}:{self.name_suffix}:{key}")
            _jitter_target(node, rng, self.ra_spread, self.dec_spread)
            self.jittered[key] = (copy.deepcopy(sidereal.get("ra")), copy.deepcopy(sidereal.get("dec")))
        else:
            # Copied, not recomputed from the rounded strings, so every copy is identical.
            ra, dec = self.jittered[key]
            sidereal["ra"], sidereal["dec"] = copy.deepcopy(ra), copy.deepcopy(dec)
        # Sight keys targets by name; a shared name would merge the clones into one target.
        node["name"] = f"{name}{self.name_suffix}"


def _rewrite(node: Any, ctx: _CloneContext) -> None:
    if isinstance(node, dict):
        if node.get("sidereal") and isinstance(node.get("name"), str):
            ctx.target(node)
        for key, value in node.items():
            if isinstance(value, str):
                node[key] = ctx.map_str(value)
            elif isinstance(value, (dict, list)):
                _rewrite(value, ctx)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            if isinstance(value, str):
                node[i] = ctx.map_str(value)
            elif isinstance(value, (dict, list)):
                _rewrite(value, ctx)


def _clone_label(old_label: str, number: int) -> str:
    match = _PROGRAM_LABEL.match(old_label)
    return f"{match.group(1)}-{number}{match.group(3) or ''}"


def make_clone(raw: dict, j: int, seed: int,
               ra_spread: float = 45.0, dec_spread: float = 5.0) -> Tuple[dict, str]:
    """A deep copy of one fetched program with new ids, labels, target names and positions.

    Every id is rewritten through one mapping, so cross references (parent ids, group
    ids, target ids) stay consistent inside the clone. The suffix has a fixed width, so
    two different (id, clone) pairs can never produce the same id.
    """
    out = copy.deepcopy(raw)
    body = _program_body(out)
    old_label = body["reference"]["label"]
    new_label = _clone_label(old_label, CLONE_NUMBER_BASE + j)
    ids: set = set()
    _collect_internal_ids(body, ids)
    suffix = f"f{j:05x}"
    ctx = _CloneContext(id_map={i: i + suffix for i in ids},
                        old_label=old_label, new_label=new_label,
                        name_suffix=f"~s{j}", seed=str(seed),
                        ra_spread=ra_spread, dec_spread=dec_spread)
    _rewrite(body, ctx)
    return out, new_label


@dataclass
class _Source:
    index: int
    label: str
    raw: dict
    counts: Dict[str, int]
    # The program's active dates, as the aggregator's program_window gives them.
    window: Tuple[date, date] = (date.min, date.max)


@dataclass
class CloneSet:
    sources: List[_Source]
    excluded: Counter
    queue: List[Tuple[int, int]]                  # (clone number j, source index)
    selections: Dict[int, List[int]]              # size -> clone numbers
    totals: Dict[int, Dict[str, int]]             # size -> observations per site
    payloads: Dict[int, dict]                     # clone number -> payload
    obs_ids: Dict[int, List[str]]                 # clone number -> observation labels
    obs_with_target: Dict[int, List[str]]         # the ones Sight stores visibility for
    target_names: Dict[int, List[str]]            # clone number -> base target names


def _provider():
    from lucupy.minimodel import ObservationClass
    from scheduler.core.programprovider.gpp import GppProgramProvider
    from scheduler.core.sources import Origins
    from scheduler.core.sources.sources import Sources

    sources = Sources()
    sources.set_origin(Origins.OPS())
    return GppProgramProvider(frozenset({ObservationClass.SCIENCE, ObservationClass.PROGCAL,
                                         ObservationClass.PARTNERCAL}), sources)


def _load_sources(fetched: dict, night_start: datetime, allow_nonsidereal: bool,
                  provider) -> Tuple[List[_Source], Counter]:
    """Captured programs that parse, are active on the night, and have observations."""
    from lucupy.minimodel import NonsiderealTarget
    from lucupy.types import ZeroTime

    night = night_start.replace(tzinfo=None)
    sources: List[_Source] = []
    excluded: Counter = Counter()
    for raw in fetched["programs"]:
        body = _program_body(raw)
        label = (body.get("reference") or {}).get("label")
        if not label or not _PROGRAM_LABEL.match(str(label)):
            excluded["no science reference label"] += 1
            continue
        try:
            program = provider.parse_program(body)
        except Exception:
            excluded["fails to parse"] += 1
            continue
        if program is None:
            excluded["parses to nothing"] += 1
            continue
        if program.program_awarded() == ZeroTime:
            excluded["no awarded time"] += 1
            continue
        if not (program.start <= night <= program.end):
            excluded["not active on the night"] += 1
            continue
        observations = list(program.observations())
        if not observations:
            excluded["no observations"] += 1
            continue
        # Non-sidereal targets make local visibility call JPL Horizons over the network.
        if not allow_nonsidereal and any(isinstance(o.base_target(), NonsiderealTarget)
                                         for o in observations):
            excluded["non-sidereal target"] += 1
            continue
        counts = Counter(o.site.name for o in observations if o.site is not None)
        if not counts:
            excluded["no site"] += 1
            continue
        sources.append(_Source(len(sources), str(label), raw, dict(counts),
                               (program.start.date(), program.end.date())))
    return sources, excluded


def _build_queue(sources: List[_Source], cap: int) -> List[Tuple[int, int]]:
    """Every clone any size up to `cap` may use, in a fixed order.

    Depends only on the fetched programs and the cap, never on the sizes asked for, so the clone
    numbers (and so the labels) are the same in seed-sight and in every run.
    """
    sites = sorted({site for s in sources for site in s.counts})
    biggest = {site: max(s.counts.get(site, 0) for s in sources) for site in sites}
    # Headroom past the cap so the exact fill below still finds small programs.
    want = {site: cap + 2 * biggest[site] for site in sites}
    supplied = {site: 0 for site in sites}
    queue: List[Tuple[int, int]] = []
    j = 0
    while any(supplied[site] < want[site] for site in sites):
        for source in sources:
            if all(supplied[site] >= want[site] for site in source.counts):
                continue
            queue.append((j, source.index))
            for site, n in source.counts.items():
                supplied[site] += n
            j += 1
    return queue


def _select(queue: List[Tuple[int, int]], sources: List[_Source], sites: List[str],
            size: int) -> Tuple[List[int], Dict[str, int]]:
    """Clones that bring every site to `size` observations, exactly when possible."""
    source_of = dict(queue)
    totals = {site: 0 for site in sites}
    taken: List[int] = []
    taken_set: set = set()
    for j, index in queue:
        counts = sources[index].counts
        if not any(counts.get(site, 0) and totals[site] < size for site in sites):
            continue
        if any(totals[site] + counts.get(site, 0) > size for site in sites):
            continue
        taken.append(j)
        taken_set.add(j)
        for site, n in counts.items():
            totals[site] += n
        if all(totals[site] >= size for site in sites):
            return taken, totals
    # No exact fit left: top up, each time with the clone that overshoots the least.
    while not all(totals[site] >= size for site in sites):
        best: Optional[Tuple[int, int]] = None
        for j, index in queue:
            counts = sources[index].counts
            if j in taken_set or not any(counts.get(site, 0) and totals[site] < size for site in sites):
                continue
            overshoot = sum(max(0, totals[site] + counts.get(site, 0) - size) for site in sites)
            if best is None or overshoot < best[0]:
                best = (overshoot, j)
        if best is None:
            break
        j = best[1]
        taken.append(j)
        taken_set.add(j)
        for site, n in sources[source_of[j]].counts.items():
            totals[site] += n
    return taken, totals


def build_clone_set(fetched: dict, night_start: datetime, sizes: List[int], cap: int,
                    seed: int, allow_nonsidereal: bool) -> CloneSet:
    provider = _provider()
    sources, excluded = _load_sources(fetched, night_start, allow_nonsidereal, provider)
    if not sources:
        sys.exit(f"No usable source programs in the ODB for {night_start.date()}: {dict(excluded)}")
    sites = sorted({site for s in sources for site in s.counts})
    for site in ("GN", "GS"):
        if site not in sites:
            sys.exit(f"The ODB has no usable programs at {site}; realtime plans both sites.")

    queue = _build_queue(sources, cap)
    selections: Dict[int, List[int]] = {}
    totals: Dict[int, Dict[str, int]] = {}
    for size in sizes:
        selections[size], totals[size] = _select(queue, sources, sites, size)

    source_of = dict(queue)
    payloads: Dict[int, dict] = {}
    obs_ids: Dict[int, List[str]] = {}
    obs_with_target: Dict[int, List[str]] = {}
    target_names: Dict[int, List[str]] = {}
    seen: Dict[str, int] = {}
    for j in sorted({j for chosen in selections.values() for j in chosen}):
        source = sources[source_of[j]]
        payload, new_label = make_clone(source.raw, j, seed)
        # Self-check: the clone parses to what its source parsed to, under its new name.
        program = provider.parse_program(_program_body(payload))
        if program is None or program.id.id != new_label:
            sys.exit(f"Clone {j} of {source.label} did not parse as {new_label}.")
        observations = list(program.observations())
        counts = dict(Counter(o.site.name for o in observations if o.site is not None))
        if counts != source.counts:
            sys.exit(f"Clone {new_label} has {counts} observations per site, its source "
                     f"{source.label} has {source.counts}.")
        for o in observations:
            if o.id.id in seen:
                sys.exit(f"Observation id {o.id.id} appears in clones {seen[o.id.id]} and {j}.")
            seen[o.id.id] = j
        payloads[j] = payload
        obs_ids[j] = [o.id.id for o in observations]
        obs_with_target[j] = [o.id.id for o in observations if o.base_target() is not None]
        target_names[j] = sorted({str(o.base_target().name) for o in observations
                                  if o.base_target() is not None})

    return CloneSet(sources, excluded, queue, selections, totals, payloads, obs_ids, obs_with_target,
                    target_names)


def _print_clone_summary(clones: CloneSet) -> None:
    print(f"Sources: {len(clones.sources)} programs usable"
          + (f", excluded {dict(clones.excluded)}" if clones.excluded else ""))
    for size, totals in clones.totals.items():
        print(f"  size {size}: {len(clones.selections[size])} clones, "
              + ", ".join(f"{site} {n}" for site, n in sorted(totals.items())))


# --------------------------------------------------------------------------------------
# Instrumentation (harness only)
# --------------------------------------------------------------------------------------

class _Recorder:
    """Per-plan timings, hot-path totals and funnel counts, keyed by run id."""

    def __init__(self) -> None:
        self.events: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self.hot: Dict[str, Dict[str, List[float]]] = defaultdict(dict)
        self.funnel: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(dict)
        self.parsed_ids: Dict[str, List[str]] = {}

    @staticmethod
    def run_id() -> str:
        from scheduler.context import schedule_id_var
        return schedule_id_var.get()

    def add_hot(self, name: str, seconds: float) -> None:
        entry = self.hot[self.run_id()].setdefault(name, [0, 0.0, 0.0])
        entry[0] += 1
        entry[1] += seconds
        entry[2] = max(entry[2], seconds)

    def set_funnel(self, stage: str, counts: Dict[str, int]) -> None:
        self.funnel[self.run_id()][stage] = counts


class _PerfCapture(logging.Handler):
    """The scheduler.perf logger does not propagate, so it needs its own handler."""

    def __init__(self, recorder: _Recorder) -> None:
        super().__init__()
        self._recorder = recorder

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = json.loads(record.getMessage())
        except Exception:
            return
        if "duration_s" in payload:
            self._recorder.events[payload.get("run_id", "?")].append(
                (payload["event"], float(payload["duration_s"])))


class _Patches:
    def __init__(self) -> None:
        self._undo: List[Callable[[], None]] = []

    @staticmethod
    def _raw(owner: type, name: str) -> Any:
        for klass in owner.__mro__:
            if name in klass.__dict__:
                return klass.__dict__[name]
        raise AttributeError(f"{owner.__name__}.{name}")

    def wrap(self, owner: type, name: str, make: Callable[[Callable], Callable]) -> None:
        """Replace owner.name with make(original function), keeping static/classmethod."""
        raw = self._raw(owner, name)
        kind = type(raw) if isinstance(raw, (staticmethod, classmethod)) else None
        function = raw.__func__ if kind else raw
        replacement = make(function)
        if name in owner.__dict__:
            self._undo.append(lambda: setattr(owner, name, raw))
        else:
            self._undo.append(lambda: delattr(owner, name))
        setattr(owner, name, kind(replacement) if kind else replacement)

    def set(self, module: Any, name: str, value: Any) -> None:
        original = getattr(module, name)
        self._undo.append(lambda: setattr(module, name, original))
        setattr(module, name, value)

    def undo(self) -> None:
        while self._undo:
            self._undo.pop()()


def _timed_wrapper(operation: str, after: Optional[Callable[[Any], None]] = None):
    from scheduler.services.telemetry import timed

    def make(function: Callable) -> Callable:
        if inspect.iscoroutinefunction(function):
            @functools.wraps(function)
            async def async_wrapper(*args, **kwargs):
                with timed(operation):
                    result = await function(*args, **kwargs)
                _safe(after, result)
                return result
            return async_wrapper

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            with timed(operation):
                result = function(*args, **kwargs)
            _safe(after, result)
            return result
        return wrapper
    return make


def _safe(after: Optional[Callable[[Any], None]], result: Any) -> None:
    """Funnel bookkeeping is the harness's business; it must never fail a plan."""
    if after is None:
        return
    try:
        after(result)
    except Exception:
        pass


def _hot_wrapper(name: str, recorder: _Recorder):
    """count + total only: one JSON line per call on a hot path would skew the result."""
    def make(function: Callable) -> Callable:
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                recorder.add_hot(name, time.perf_counter() - started)
        return wrapper
    return make


def _after_wrapper(after: Callable[[Any], None]):
    def make(function: Callable) -> Callable:
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            result = function(*args, **kwargs)
            try:
                after(result)
            except Exception:
                pass
            return result
        return wrapper
    return make


def _install_instrumentation(patches: _Patches, recorder: _Recorder, live: bool) -> None:
    from lucupy.minimodel import NightIndex
    from scheduler.core.components.collector import Collector
    from scheduler.core.components.optimizer.greedymax import GreedyMaxOptimizer
    from scheduler.core.components.ranker import DefaultRanker
    from scheduler.core.components.selector import Selector
    from scheduler.core.scp.scp import SCP
    from scheduler.core.statscalculator import StatCalculator
    from scheduler.engine.engineRT import EngineRT
    from scheduler.graphql_mid.types import SNightTimelines, SRunSummary

    def by_site(observations: Iterable) -> Dict[str, int]:
        return dict(Counter(o.site.name for o in observations if o.site is not None))

    def parsed(result):
        parsed_observations, _bad = result
        recorder.set_funnel("parsed", by_site(obs for _p, obs in parsed_observations))
        recorder.parsed_ids[recorder.run_id()] = [obs.id.id for _p, obs in parsed_observations
                                                  if obs.base_target() is not None]

    def night_filtered(result):
        recorder.set_funnel("night-0 resources", by_site(result.get(NightIndex(0), [])))

    def selected(selection):
        unique = {}
        for group_data in selection.schedulable_groups.values():
            for obs in group_data.group.observations():
                unique[obs.id] = obs
        recorder.set_funnel("in selection", by_site(unique.values()))

    def planned(plans):
        recorder.set_funnel("scheduled visits",
                            {site.name: len(plan.visits) for site, plan in plans.plans.items()})

    if live:
        # The bulk get_all behind gpp_program_data: the network part of the collector build.
        from gpp_client.domains.scheduler import SchedulerDomain

        def fetched(programs):
            recorder.set_funnel("payload", {k: v for k, v in _raw_site_counts(programs).items()
                                            if k != "unknown"})

        patches.wrap(SchedulerDomain, "get_all", _timed_wrapper("stress.odb_fetch", fetched))
    patches.wrap(Collector, "async_init_night_events", _timed_wrapper("stress.night_events"))
    patches.wrap(Collector, "_parse_programs", _timed_wrapper("stress.parse_programs", parsed))
    patches.wrap(Collector, "_filter_by_night_configuration",
                 _timed_wrapper("stress.night_filter", night_filtered))
    patches.wrap(Collector, "night_configurations",
                 _hot_wrapper("collector.night_configurations", recorder))
    patches.wrap(Selector, "__post_init__", _timed_wrapper("stress.selector_init"))
    patches.wrap(Selector, "select", _after_wrapper(selected))
    patches.wrap(DefaultRanker, "__init__", _timed_wrapper("stress.ranker_init"))
    patches.wrap(EngineRT, "init_variant", _timed_wrapper("stress.init_variant"))
    patches.wrap(SCP, "run_rt", _after_wrapper(planned))
    patches.wrap(GreedyMaxOptimizer, "_find_max_group", _hot_wrapper("greedymax.find_max_group", recorder))
    patches.wrap(GreedyMaxOptimizer, "_update_score", _hot_wrapper("greedymax.update_score", recorder))
    patches.wrap(GreedyMaxOptimizer, "_add_visit", _hot_wrapper("greedymax.add_visit", recorder))
    patches.wrap(StatCalculator, "calculate_stitched_timeline_stats", _timed_wrapper("stress.stats"))
    patches.wrap(SNightTimelines, "from_computed_stitched_timelines",
                 _timed_wrapper("stress.serialize_timelines"))
    patches.wrap(SRunSummary, "from_computed_run_summary", _timed_wrapper("stress.serialize_summary"))


def _setup_metrics():
    """In-memory reader always; OTLP as well when an endpoint is configured."""
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader, PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from scheduler.services.telemetry import otel

    reader = InMemoryMetricReader()
    readers = [reader]
    resource = otel.build_resource()
    exporting = bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"))
    if exporting:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        readers.append(PeriodicExportingMetricReader(OTLPMetricExporter()))
        # Merged last so it wins over OTEL_RESOURCE_ATTRIBUTES: stress numbers must never
        # land in a real environment's panels.
        resource = resource.merge(Resource.create({"deployment.environment": "stress"}))
    provider = MeterProvider(resource=resource, metric_readers=readers,
                             views=otel.duration_views())
    otel.setup_telemetry(meter_provider=provider, set_global=False)
    return reader, exporting


def _sight_fallbacks(reader) -> int:
    from scheduler.services.telemetry.instruments import SIGHT_FALLBACK
    data = reader.get_metrics_data()
    if data is None:
        return 0
    return int(sum(point.value
                   for resource_metric in data.resource_metrics
                   for scope_metric in resource_metric.scope_metrics
                   for metric in scope_metric.metrics
                   if metric.name == SIGHT_FALLBACK
                   for point in metric.data.data_points))


# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------

class _FakeWeather:
    """Stands in for WeatherEventSource.get_current_state."""

    def __init__(self, iq: float, cc: float) -> None:
        self._state = [{"site": site, "imageQuality": iq, "cloudCover": cc,
                        "windDirection": 0.0, "windSpeed": 0.0} for site in ("GN", "GS")]

    async def get_current_state(self):
        return copy.deepcopy(self._state)


@dataclass
class _PlanResult:
    run_id: str
    event_time: Optional[datetime]
    ok: bool
    error: Optional[str]
    peak_bytes: Optional[int]
    worst_stall: float


@dataclass
class _SizeResult:
    size: int
    expected: Dict[str, int]
    plans: List[_PlanResult]
    fallbacks: int


async def _check_sight_coverage(clones: CloneSet, night_start: datetime,
                                vis_end: datetime) -> Dict[str, Any]:
    """Refuse to measure Sight against nights or clones it was never given.

    An observation is due on a night when the night is inside its program's window
    widened to the first 15 nights, which is exactly what seed-sight --window full
    stores. Nights past a program's window have no rows in production either.

    Returns what the DB holds for the clones, for the report.
    """
    from sqlalchemy import distinct, func, select
    from scheduler.services.sight.calculator.calculator import Calculator
    from scheduler.services.sight.database.connection import session_scope
    from scheduler.services.sight.database.models import VisibilityData

    source_of = dict(clones.queue)
    narrow_end = night_start.date() + timedelta(days=NIGHTS_IN_WINDOW - 1)
    # Observations without a base target never get a Sight row, so they can't be "missing".
    due_window: Dict[str, Tuple[date, date]] = {}
    for j in {j for chosen in clones.selections.values() for j in chosen}:
        start, end = clones.sources[source_of[j]].window
        window = (min(start, night_start.date()), max(end, narrow_end))
        for oid in clones.obs_with_target[j]:
            due_window[oid] = window

    nights = _window_nights(night_start, vis_end)
    selected = {j for chosen in clones.selections.values() for j in chosen}
    names = sorted({name for j in selected for name in clones.target_names[j]})
    print(f"Checking Sight coverage: {len(names)} targets, {len(nights)} nights "
          f"({nights[0]}..{nights[-1]})...")
    async with session_scope() as session:
        calc = Calculator(session)
        # Clones are rebuilt from the live ODB on every command. A target name carries its
        # source and clone number, so a missing one means the ODB moved since seed-sight.
        found = await calc.target_repo.get_ids_by_names(names)
        missing_names = sorted(set(names) - set(found))
        if missing_names:
            sys.exit(f"Sight has no target for {len(missing_names)} of the {len(names)} clone targets "
                     f"(e.g. {missing_names[:3]}). The ODB has changed since seed-sight ran, so the "
                     f"clones changed too: run seed-sight again with the same --night/--cap/--programs.")
        for night in nights:
            due = {oid for oid, (start, end) in due_window.items() if start <= night <= end}
            if not due:
                continue
            stored = await calc.visibility_repo.get_stored_observation_ids_on_night(night)
            missing = due - stored
            if missing:
                hint = ("--window full first (a narrow seed only serves --vis-end realtime-default)"
                        if night > narrow_end
                        else "the same --night/--cap/--sizes/--programs first (the ODB may have changed)")
                sys.exit(f"Sight has no visibility on {night} for {len(missing)} of the {len(due)} "
                         f"clone observations due that night (e.g. {sorted(missing)[:3]}). Run "
                         f"seed-sight with {hint}.")
        first, last, count = (await session.execute(
            select(func.min(VisibilityData.night_date), func.max(VisibilityData.night_date),
                   func.count(distinct(VisibilityData.night_date)))
            .where(VisibilityData.observation_id.in_(sorted(due_window))))).one()
    return {"db_first": first, "db_last": last, "db_nights": count}


async def _measure_size(size: int, payload: Optional[List[dict]], expected: Dict[str, int], plans: int,
                        offset_hours: float, night_start: datetime, weather: _FakeWeather,
                        current: dict, recorder: _Recorder, reader, run_prefix: str) -> _SizeResult:
    from lucupy.minimodel import ALL_SITES, NightIndex
    from scheduler.context import schedule_id_var
    from scheduler.core.builder.modes import SchedulerModes
    from scheduler.core.events.queue import NightlyTimelineStore
    from scheduler.core.events.queue.events import OnDemandScheduleEvent
    from scheduler.engine.engineRT import EngineRT
    from scheduler.engine.params import SchedulerParameters
    from scheduler.services.loop_monitor import LoopMonitor
    from scheduler.services.telemetry import read_memory

    # The realtime defaults (default_operation_parameters). A longer window comes from
    # BuildParameters.visibility_end, set in _run, exactly as an operator would set it.
    params = SchedulerParameters(start=night_start,
                                 end=night_start + timedelta(days=NIGHTS_IN_WINDOW - 1),
                                 sites=ALL_SITES,
                                 mode=SchedulerModes.OPERATION,
                                 semester_visibility=False,
                                 num_nights_to_schedule=1)
    # A fresh store per size: the plan history of one size must not stitch into the next.
    engine = EngineRT(params, None, "stress", weather, NightlyTimelineStore())
    fallbacks_before = _sight_fallbacks(reader)
    earliest_start: Optional[datetime] = None
    results: List[_PlanResult] = []

    for i in range(plans):
        run_id = f"{run_prefix}-{size}-{i + 1}"
        event_time = None
        if i > 0 and earliest_start is not None:
            event_time = earliest_start + timedelta(hours=offset_hours * i)
        # Copied here, outside every timer: parsing must never see a payload an earlier
        # plan touched, and a copy inside the build would be billed to the collector.
        if payload is not None:
            current["items"] = copy.deepcopy(payload)
        event = OnDemandScheduleEvent(site=None, description=f"stress {run_id}", time=event_time)

        token = schedule_id_var.set(run_id)
        monitor = LoopMonitor()
        monitor.start()
        ok, error = True, None
        try:
            await engine.compute_event_plan(event)
        except Exception as exc:
            ok, error = False, f"{type(exc).__name__}: {_short(exc)}"
            traceback.print_exc()
        finally:
            await monitor.stop()
            current["items"] = []
            if engine.scp is not None:
                collector = engine.scp.collector
                visible = collector.get_visible_observations_for_night(NightIndex(0))
                sites = Counter()
                for oid in visible:
                    obs = collector.get_observation(oid)
                    if obs is not None and obs.site is not None:
                        sites[obs.site.name] += 1
                recorder.set_funnel("visible night 0", dict(sites))
                if earliest_start is None:
                    starts = [collector.get_night_length(site, NightIndex(0))[0] for site in ALL_SITES]
                    earliest_start = min(starts)
            schedule_id_var.reset(token)

        reading = read_memory()
        results.append(_PlanResult(run_id, event_time, ok, error, reading.peak_bytes,
                                   monitor.worst_stall))
        plan_time = next((d for e, d in recorder.events.get(run_id, []) if e == "engine.plan"), None)
        print(f"  {run_id}: {'ok' if ok else 'FAILED'}"
              + (f" in {plan_time:.1f}s" if plan_time is not None else "")
              + (f" ({error})" if error else ""))

    return _SizeResult(size, expected, results, _sight_fallbacks(reader) - fallbacks_before)


async def _live_sight_coverage(ids: List[str], night: date) -> Dict[str, Any]:
    """How many of the parsed observations Sight has a row for on night 0.

    Live data is whatever the aggregator last stored, so this is reported rather than
    enforced: a missing row reads as "not visible" and shrinks the retrieval work.
    """
    from scheduler.services.sight.calculator.calculator import Calculator
    from scheduler.services.sight.database.connection import session_scope

    async with session_scope() as session:
        stored = await Calculator(session).visibility_repo.get_stored_observation_ids_on_night(night)
    return {"night": night, "parsed": len(ids), "stored": len(set(ids) & stored)}


async def _run(args) -> int:
    from lucupy.observatory.abstract import ObservatoryProperties
    from lucupy.observatory.gemini.geminiproperties import GeminiProperties
    from scheduler.engine.params import BuildParameters, build_params_store
    from scheduler.services.telemetry import otel
    from scheduler.services.telemetry.perf import PERF_LOGGER_NAME

    ObservatoryProperties.set_properties(GeminiProperties)
    live = args.source == "odb"
    night_start = _night_anchor(args.night)
    vis_end = _vis_end(night_start, args.vis_end)
    nights = len(_window_nights(night_start, vis_end))
    _check_ops_calendar(night_start, vis_end)
    print(f"Engine reads {nights} nights: {night_start.date()}..{vis_end.date()}.")

    meta: dict = {}
    clones: Optional[CloneSet] = None
    if live:
        sizes = [LIVE]
    else:
        sizes = _parse_sizes(args.sizes or DEFAULT_SIZES, args.cap)
        sources = await _fetch_sources(args.programs)
        meta = sources["meta"]
        print(f"Building clones for {night_start.date()} (cap {args.cap} per site)...")
        clones = build_clone_set(sources, night_start, sizes, args.cap, args.seed, args.allow_nonsidereal)
        _print_clone_summary(clones)

    sight = args.visibility == "sight"
    coverage: Optional[Dict[str, Any]] = None
    if sight:
        from scheduler.services.sight.database.connection import dispose_engine, init_db_engine
        await init_db_engine()
        if clones is not None:
            coverage = await _check_sight_coverage(clones, night_start, vis_end)

    program_ids: Optional[List[str]] = None
    if live and args.programs:
        program_ids = await _resolve_program_ids(args.programs)

    current: dict = {"items": []}

    async def fake_gpp_program_data(program_list=None):
        async def generate():
            for item in current["items"]:
                yield item
        return generate()

    import scheduler.core.builder.simulationbuilder as simulationbuilder
    recorder = _Recorder()
    patches = _Patches()
    perf_logger = logging.getLogger(PERF_LOGGER_NAME)
    saved_handlers = list(perf_logger.handlers)
    capture_handler = _PerfCapture(recorder)
    reader, exporting = _setup_metrics()
    weather = _FakeWeather(args.iq, args.cc)
    started_at = datetime.now(UTC)
    results: List[_SizeResult] = []

    try:
        if not live:
            # The realtime loading seam (simulationbuilder.py: gpp_program_data -> async_load_programs).
            # Live runs leave it alone, so every plan fetches from the ODB as realtime does.
            patches.set(simulationbuilder, "gpp_program_data", fake_gpp_program_data)
        _install_instrumentation(patches, recorder, live)
        # After the imports above: each module logger is created, with its own level, on import.
        _quiet_scheduler_logs(args.log_level)
        # JSON lines go to the recorder instead of the console.
        for handler in saved_handlers:
            perf_logger.removeHandler(handler)
        perf_logger.addHandler(capture_handler)

        # How an operator widens a realtime run: EngineRT.build reads these over the defaults.
        await build_params_store.set(BuildParameters(visibility_start=night_start,
                                                     visibility_end=vis_end,
                                                     program_list=program_ids))

        def payload_for(size: int) -> Optional[List[dict]]:
            return None if clones is None else [clones.payloads[j] for j in clones.selections[size]]

        def expected_for(size: int) -> Dict[str, int]:
            return {} if clones is None else clones.totals[size]

        smallest = min(sizes)
        if not args.no_warmup:
            print("Warm-up plan (not measured)...")
            await _measure_size(smallest, payload_for(smallest), expected_for(smallest), 1,
                                args.offset_hours, night_start, weather, current, recorder,
                                reader, "warmup")

        for size in sizes:
            print(f"{'Live ODB data' if live else f'Size {size} per site ({clones.totals[size]})'}, "
                  f"{args.plans} plan(s)...")
            results.append(await _measure_size(size, payload_for(size), expected_for(size),
                                               args.plans, args.offset_hours, night_start, weather,
                                               current, recorder, reader, "live" if live else "stress"))

        if live and sight and results and results[0].plans:
            # After every measured plan, so it can't touch the numbers.
            ids = recorder.parsed_ids.get(results[0].plans[0].run_id, [])
            coverage = await _live_sight_coverage(ids, night_start.date())
    finally:
        patches.undo()
        perf_logger.removeHandler(capture_handler)
        for handler in saved_handlers:
            perf_logger.addHandler(handler)
        # Flushes the OTLP exporter when there is one.
        otel.shutdown_telemetry()
        from scheduler.clients.gpp import gpp
        # The client holds sockets bound to this loop; close them before it ends.
        await gpp.close()
        if sight:
            await dispose_engine()

    section = _render_section(args, night_start, vis_end, coverage, meta, clones, program_ids,
                              results, recorder, started_at, exporting)
    _write_report(args.label, section)
    print(f"\nReport: {REPORT_PATH.relative_to(REPO)} (section '{args.label}')")
    return 0 if all(p.ok for r in results for p in r.plans) else 1


def _parse_sizes(text: str, cap: int) -> List[int]:
    try:
        sizes = sorted({int(s) for s in text.split(",") if s.strip()})
    except ValueError:
        sys.exit(f"--sizes must be comma-separated integers, got {text!r}")
    if not sizes or sizes[0] <= 0:
        sys.exit("--sizes must be positive.")
    if sizes[-1] > cap:
        sys.exit(f"--sizes go up to {sizes[-1]} but --cap is {cap}; raise --cap (and re-seed Sight).")
    return sizes


# --------------------------------------------------------------------------------------
# seed-sight
# --------------------------------------------------------------------------------------

async def _seed_sight(args) -> int:
    from lucupy.observatory.abstract import ObservatoryProperties
    from lucupy.observatory.gemini.geminiproperties import GeminiProperties
    import scheduler.services.visibility_aggregator.aggregator as aggregator
    from scheduler.config import config
    from scheduler.services.sight.calculator.calculator import Calculator
    from scheduler.services.sight.database.connection import (dispose_engine, init_db_engine,
                                                               session_scope)
    from scheduler.services.telemetry.perf import PERF_LOGGER_NAME

    ObservatoryProperties.set_properties(GeminiProperties)
    _quiet_scheduler_logs("WARNING")
    logging.getLogger(aggregator.__name__).setLevel(logging.INFO)
    sizes = _parse_sizes(args.sizes or DEFAULT_SIZES, args.cap)
    night_start = _night_anchor(args.night)
    _check_ops_calendar(night_start)

    try:
        sources = await _fetch_sources(args.programs)
    finally:
        from scheduler.clients.gpp import gpp
        # The client holds sockets bound to this loop; close them before it ends.
        await gpp.close()
    print(f"Building clones for {night_start.date()} (cap {args.cap} per site)...")
    clones = build_clone_set(sources, night_start, sizes, args.cap, args.seed, args.allow_nonsidereal)
    _print_clone_summary(clones)
    payload = [clones.payloads[j] for j in sorted(clones.payloads)]

    async def fake_gpp_program_data(program_list=None):
        async def generate():
            for item in payload:
                yield item
        return generate()

    patches = _Patches()
    patches.set(aggregator, "gpp_program_data", fake_gpp_program_data)
    # Seeding is setup, not a measurement: keep the perf JSON lines off the console.
    perf_logger = logging.getLogger(PERF_LOGGER_NAME)
    saved_handlers = list(perf_logger.handlers)
    for handler in saved_handlers:
        perf_logger.removeHandler(handler)
    perf_logger.addHandler(logging.NullHandler())

    narrow = (night_start.date(), night_start.date() + timedelta(days=NIGHTS_IN_WINDOW - 1))
    await init_db_engine()
    try:
        targets_by_name, requests, windows, _labels, counts = await aggregator._collect_requests([])
        if args.window == "narrow":
            windows = {oid: narrow for oid in windows}
        else:
            # The program's own window, widened to always hold the nights the engine reads.
            windows = {oid: (min(w[0], narrow[0]), max(w[1], narrow[1])) for oid, w in windows.items()}
        print(f"{len(targets_by_name)} targets, {len(requests)} observations, window={args.window} "
              f"({min(w[0] for w in windows.values())}..{max(w[1] for w in windows.values())}); "
              f"{counts['skipped_no_target']} without a usable target.")

        started_at = datetime.now(UTC).isoformat()
        batch_size = max(1, int(config.visibility_aggregator.target_batch_size))
        async with session_scope() as session:
            calc = Calculator(session)
            windows_by_target = aggregator.target_windows(requests, windows)
            groups: Dict[Tuple[date, date], list] = {}
            for target in targets_by_name.values():
                window = windows_by_target.get(target.name)
                if window is not None:
                    groups.setdefault(window, []).append(target)
            total = sum(len(g) for g in groups.values())
            done = 0
            for (window_start, window_end), group in sorted(groups.items()):
                for offset in range(0, len(group), batch_size):
                    chunk = group[offset:offset + batch_size]
                    await calc.create_targets_bulk(chunk, window_start, window_end)
                    await calc.precompute_stage1(window_start, window_end,
                                                 target_names=[t.name for t in chunk])
                    await session.commit()
                    done += len(chunk)
                    print(f"  Stage 1: {done}/{total} targets")
            stored = await aggregator._store_missing_visibility(calc, requests, windows, None, started_at)
        print(f"Seeded: {stored} new visibility rows. Clone labels start at G-....-{CLONE_NUMBER_BASE}.")
    finally:
        patches.undo()
        perf_logger.handlers.clear()
        for handler in saved_handlers:
            perf_logger.addHandler(handler)
        await dispose_engine()
    return 0


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------

# (operation, depth, description). Depth only drives indentation in the table.
_ROWS = [
    ("engine.plan", 0, "Whole realtime plan"),
    ("scp.build_collector", 1, "Collector build"),
    ("stress.odb_fetch", 2, "ODB fetch (live only)"),
    ("stress.night_events", 2, "Night events"),
    ("stress.parse_programs", 2, "Parse programs"),
    ("stress.night_filter", 2, "Night configuration filter"),
    ("collector.visibility_local", 2, "Visibility, local compute"),
    ("collector.visible_observations", 2, "Sight: visible observations query (per night)"),
    ("collector.stage1_bulk", 2, "Sight: stage 1 bulk fetch"),
    ("collector.sight_apply", 2, "Sight: build target info"),
    ("stress.selector_init", 1, "Selector setup"),
    ("stress.ranker_init", 1, "Ranker setup"),
    ("stress.init_variant", 1, "Weather variant"),
    ("scp.select", 1, "Selection"),
    ("scp.score_program", 2, "Score programs"),
    ("derived.select_overhead", 2, "Program deepcopy and the rest"),
    ("scp.optimize", 1, "GreedyMax"),
    ("greedymax.find_max_group", 2, "Find best group"),
    ("greedymax.update_score", 2, "Re-score after a visit"),
    ("greedymax.add_visit", 2, "Add visit"),
    ("stress.stats", 1, "Timeline stats"),
    ("stress.serialize_timelines", 1, "Serialize timelines"),
    ("stress.serialize_summary", 1, "Serialize summary"),
    ("collector.night_configurations", 1, "Night configurations, all callers (overlaps rows above)"),
    ("derived.untracked", 1, "Untracked: plan minus the timed steps"),
]
# Direct children of engine.plan; whatever they don't cover is "untracked".
_PLAN_CHILDREN = ("scp.build_collector", "stress.selector_init", "stress.ranker_init",
                  "stress.init_variant", "scp.select", "scp.optimize", "stress.stats",
                  "stress.serialize_timelines", "stress.serialize_summary")
_VISIBILITY_OPS = ("collector.visibility_local", "collector.visible_observations",
                   "collector.stage1_bulk", "collector.sight_apply")
_SCALING = (("engine.plan", "Whole plan"), ("scp.build_collector", "Collector build"),
            ("derived.visibility", "Visibility"), ("scp.select", "Selection"),
            ("scp.optimize", "GreedyMax"))
_FUNNEL = ("payload", "parsed", "night-0 resources", "visible night 0", "in selection",
           "scheduled visits")


def _per_plan_ops(recorder: _Recorder, run_id: str) -> Dict[str, Tuple[int, float, float]]:
    """operation -> (calls, total seconds, worst call) for one plan, derived rows included."""
    ops: Dict[str, List[float]] = defaultdict(list)
    for event, seconds in recorder.events.get(run_id, []):
        ops[event].append(seconds)
    out = {op: (len(v), sum(v), max(v)) for op, v in ops.items()}
    for op, (calls, total, worst) in recorder.hot.get(run_id, {}).items():
        out[op] = (int(calls), total, worst)

    def total(op: str) -> float:
        return out.get(op, (0, 0.0, 0.0))[1]

    if "scp.select" in out:
        overhead = max(0.0, total("scp.select") - total("scp.score_program"))
        out["derived.select_overhead"] = (1, overhead, overhead)
    if "engine.plan" in out:
        untracked = max(0.0, total("engine.plan") - sum(total(op) for op in _PLAN_CHILDREN))
        out["derived.untracked"] = (1, untracked, untracked)
    vis = sum(total(op) for op in _VISIBILITY_OPS)
    if vis:
        out["derived.visibility"] = (1, vis, vis)
    return out


def _size_label(size: int) -> str:
    return "ODB" if size == LIVE else str(size)


def _size_col(size: int) -> str:
    return "live ODB" if size == LIVE else f"{size}/site"


def _fmt_s(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    if seconds >= 100:
        return f"{seconds:.0f}s"
    if seconds >= 1:
        return f"{seconds:.1f}s"
    return f"{seconds * 1000:.0f}ms"


def _fmt_bytes(n: Optional[int]) -> str:
    return "-" if n is None else f"{n / 1024 ** 3:.2f} GB"


def _table(header: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _render_section(args, night_start: datetime, vis_end: datetime,
                    coverage: Optional[Dict[str, Any]], meta: dict, clones: Optional[CloneSet],
                    program_ids: Optional[List[str]], results: List[_SizeResult],
                    recorder: _Recorder, started_at: datetime, exporting: bool) -> str:
    sight = args.visibility == "sight"
    nights = len(_window_nights(night_start, vis_end))
    per_size: Dict[int, Dict[str, Any]] = {}
    status_notes: List[str] = []

    for result in results:
        measured = [p for p in result.plans if p.ok]
        plan_ops = [_per_plan_ops(recorder, p.run_id) for p in measured]
        plan_times = [ops.get("engine.plan", (0, None, None))[1] for ops in plan_ops]
        merged: Dict[str, List[Tuple[int, float, float]]] = defaultdict(list)
        for ops in plan_ops:
            for op, value in ops.items():
                merged[op].append(value)
        per_size[result.size] = {"result": result, "plan_ops": plan_ops, "plan_times": plan_times,
                                 "merged": merged}
        if sight:
            seen = {op for ops in plan_ops for op in ops}
            if result.fallbacks or "collector.sight_apply" not in seen or "collector.visibility_local" in seen:
                status_notes.append(f"size {_size_label(result.size)}: Sight fell back to local "
                                    f"({result.fallbacks} fallbacks recorded)")
        failed = [p for p in result.plans if not p.ok]
        if failed:
            status_notes.append(f"size {_size_label(result.size)}: {len(failed)} plan(s) failed "
                                f"({failed[0].error})")
    status = "FAILED" if status_notes else "OK"

    def mean(values: List[float]) -> Optional[float]:
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    out: List[str] = []
    out.append(f"## {args.label}\n")
    out.append(f"**Status: {status}**" + (": " + "; ".join(status_notes) if status_notes else ""))
    out.append("")
    out.append(f"- Visibility: **{args.visibility}**"
               + (" (retrieval only; filling the Sight DB is not measured)" if sight else ""))
    out.append(f"- Night: {night_start.date()} (08:00 UT anchor), both sites")
    out.append(f"- Engine reads **{nights} nights** ({night_start.date()}..{vis_end.date()})"
               + (", the default_operation_parameters window: remaining visibility, and so "
                  "scoring, sees only these nights" if args.vis_end == "realtime-default"
                  else f" through BuildParameters.visibility_end (--vis-end {args.vis_end})"))
    if coverage is not None and "db_first" in coverage:
        out.append(f"- Sight DB holds clone visibility for {coverage['db_first']}..{coverage['db_last']} "
                   f"({coverage['db_nights']} nights); every observation due on the {nights} nights "
                   f"read was present")
    elif coverage is not None:
        out.append(f"- Sight rows on {coverage['night']}: {coverage['stored']} of {coverage['parsed']} "
                   f"parsed observations with a target (the rest read as not visible)")
    out.append(f"- Run: {started_at.strftime('%Y-%m-%d %H:%M UTC')}, {args.plans} measured plan(s) per size"
               + ("" if args.no_warmup else " after one warm-up plan")
               + f"; replans at earliest evening twilight + {args.offset_hours:g}h × i")
    out.append(f"- Weather: IQ {args.iq}, CC {args.cc}, no wind (best case, so the fewest observations drop out)")
    if clones is None:
        out.append("- Data: **live ODB**, fetched inside every plan as realtime does; "
                   + (f"{len(program_ids)} programs from --programs" if program_ids
                      else "every program active today, minus the provider's exclude list"))
        out.append("- ODB fetch: included in these numbers, row `stress.odb_fetch`")
    else:
        per_site = ", ".join(f"{k} {v}" for k, v in sorted(meta.get("observations_per_site", {}).items()))
        out.append(f"- Data: clones of the live ODB, fetched at {meta.get('fetched_at')}: "
                   f"{meta.get('programs')} programs / {meta.get('observations')} observations ({per_site})"
                   + (f" from --programs {meta['program_filter']}" if meta.get("program_filter") else "")
                   + f"; {len(clones.sources)} usable as clone sources"
                   + (f", excluded {dict(clones.excluded)}" if clones.excluded else "")
                   + f"; clone seed {args.seed}, cap {args.cap}")
        out.append(f"- ODB fetch (not in these numbers): {_fmt_s(meta.get('fetch_seconds'))} for that "
                   f"fetch, once at the start of the run; the plans read the clones")
    out.append(f"- Scheduler logs at {args.log_level} (production runs at INFO)"
               + ("; metrics also pushed over OTLP as deployment.environment=stress" if exporting else ""))
    out.append("")

    # Plan time against the target.
    out.append(f"### Plan time (engine.plan) vs {PLAN_TARGET_S:.0f}s\n")
    rows = []
    for size, data in per_size.items():
        result = data["result"]
        cells = []
        times = iter(data["plan_times"])
        for plan in result.plans:
            cells.append(_fmt_s(next(times)) if plan.ok else "FAILED")
        worst = max((t for t in data["plan_times"] if t is not None), default=None)
        verdict = "-" if worst is None else ("ok" if worst <= PLAN_TARGET_S else "OVER")
        rows.append([_size_label(size)] + cells + [_fmt_s(worst), verdict])
    out.append(_table(["obs/site"] + [f"plan {i + 1}" for i in range(args.plans)] + ["max", "target"], rows))
    out.append("")

    # Scaling of the big steps.
    out.append("### Scaling (mean seconds per plan)\n")
    rows = []
    for op, title in _SCALING:
        cells = [_fmt_s(mean([v[1] for v in per_size[s]["merged"].get(op, [])])) for s in per_size]
        rows.append([title] + cells)
    out.append(_table(["step"] + [_size_col(s) for s in per_size], rows))
    out.append("")

    # Full breakdown at the largest size.
    top = max(per_size)
    merged = per_size[top]["merged"]
    plan_mean = mean([v[1] for v in merged.get("engine.plan", [])])
    out.append(f"### Breakdown {'on live ODB data' if top == LIVE else f'at {top} observations per site'} "
               f"(mean over measured plans)\n")
    rows = []
    for op, depth, description in _ROWS:
        values = merged.get(op)
        if not values:
            continue
        calls = sum(v[0] for v in values) / len(values)
        total = sum(v[1] for v in values) / len(values)
        per_call = sum(v[1] for v in values) / max(1, sum(v[0] for v in values))
        worst = max(v[2] for v in values)
        share = f"{100 * total / plan_mean:.0f}%" if plan_mean else "-"
        indent = "· " * depth
        rows.append([f"{indent}`{op}` {description}", f"{calls:.0f}", _fmt_s(total), _fmt_s(per_call),
                     _fmt_s(worst), share])
    out.append(_table(["operation", "calls/plan", "total/plan", "mean/call", "worst call", "% of plan"], rows))
    out.append("")
    out.append("`stress.*` and `greedymax.*` are timed by this harness only; `derived.*` are "
               "subtractions. Night configurations is called from inside other steps, so its "
               "share overlaps theirs.")
    out.append("")

    # Funnel per site, from the first measured plan of each size.
    out.append("### Funnel per site (first measured plan)\n")
    rows = []
    for size, data in per_size.items():
        result = data["result"]
        funnel = recorder.funnel.get(result.plans[0].run_id, {}) if result.plans else {}
        for site in ("GN", "GS"):
            # Live runs count the payload as it comes back from the ODB.
            payload = funnel.get("payload", {}).get(site, result.expected.get(site, "-"))
            cells = [str(payload)]
            for stage in _FUNNEL[1:]:
                cells.append(str(funnel.get(stage, {}).get(site, "-")))
            rows.append([_size_label(size), site] + cells)
    out.append(_table(["obs/site", "site"] + list(_FUNNEL), rows))
    out.append("")

    # Memory and event loop.
    out.append("### Memory and event loop\n")
    rows = []
    for size, data in per_size.items():
        plans = data["result"].plans
        peak = max((p.peak_bytes for p in plans if p.peak_bytes is not None), default=None)
        stall = max((p.worst_stall for p in plans), default=0.0)
        rows.append([_size_label(size), _fmt_bytes(peak), _fmt_s(stall)])
    out.append(_table(["obs/site", "peak RSS (process high-water)", "worst loop stall"], rows))
    out.append("")

    data_comment = {
        "label": args.label,
        "visibility": args.visibility,
        "status": status,
        "night": night_start.date().isoformat(),
        "nights": nights,
        "sizes": {str(size): {"plan_max": max((t for t in d["plan_times"] if t is not None), default=None),
                              "plan_mean": mean(d["plan_times"]),
                              "visibility": mean([v[1] for v in d["merged"].get("derived.visibility", [])])}
                  for size, d in per_size.items()},
    }
    out.append(f"<!-- stress-data {json.dumps(data_comment)} -->")
    return "\n".join(out)


_SECTION = re.compile(r"<!-- section:(?P<label>[A-Za-z0-9_.-]+) -->\n(?P<body>.*?)\n<!-- /section:(?P=label) -->",
                      re.DOTALL)
_DATA = re.compile(r"<!-- stress-data (\{.*?\}) -->")


def _write_report(label: str, section: str) -> None:
    sections: Dict[str, str] = {}
    if REPORT_PATH.exists():
        for match in _SECTION.finditer(REPORT_PATH.read_text()):
            sections[match.group("label")] = match.group("body")
    sections[label] = section

    data = []
    for body in sections.values():
        match = _DATA.search(body)
        if match:
            try:
                data.append(json.loads(match.group(1)))
            except json.JSONDecodeError:
                pass

    out = ["# Realtime stress test", "",
           f"_Updated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}. "
           f"Target: a plan in under {PLAN_TARGET_S:.0f}s at 1500 observations per site._", ""]
    if data:
        sizes = sorted({int(s) for d in data for s in d["sizes"]})
        out.append("## Comparison\n")
        out.append("Worst `engine.plan` per size (seconds):\n")
        rows = []
        for size in sizes:
            cells = []
            for d in data:
                value = d["sizes"].get(str(size), {}).get("plan_max")
                mark = "" if value is None or value <= PLAN_TARGET_S else " OVER"
                cells.append(_fmt_s(value) + mark)
            rows.append([_size_col(size)] + cells)
        out.append(_table(["obs/site"] + [f"{d['label']} ({d['status']}, {d.get('nights', NIGHTS_IN_WINDOW)} nights)"
                                          for d in data], rows))
        out.append("")
        out.append("Visibility step per plan (local: compute; sight: retrieval only):\n")
        rows = [[_size_col(size)] + [_fmt_s(d["sizes"].get(str(size), {}).get("visibility")) for d in data]
                for size in sizes]
        out.append(_table(["obs/site"] + [d["label"] for d in data], rows))
        out.append("")
    for name, body in sections.items():
        out.append(f"<!-- section:{name} -->\n{body}\n<!-- /section:{name} -->")
        out.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(out))


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def _add_clone_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--night", help="UT date of the 08:00 anchor, i.e. the morning the night "
                                        "ends (YYYY-MM-DD). Defaults to tonight, as realtime does.")
    parser.add_argument("--sizes",
                        help=f"Observations per site, comma-separated (default {DEFAULT_SIZES}). "
                             f"Clones only.")
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP,
                        help=f"Largest size any run will use; fixes the clone numbering "
                             f"(default {DEFAULT_CAP}). Keep it the same for seed-sight and run.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for target positions.")
    parser.add_argument("--programs", help="Comma-separated reference labels or ODB ids to fetch. "
                                           "Defaults to every program active today, as realtime.")
    parser.add_argument("--allow-nonsidereal", action="store_true",
                        help="Keep programs with non-sidereal targets. Local visibility then calls "
                             "JPL Horizons over the network.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    seed = sub.add_parser("seed-sight", help="Fill the local Sight DB with the clones (setup, not measured).")
    _add_clone_args(seed)
    seed.add_argument("--window", choices=("narrow", "full"), default="narrow",
                      help="narrow: the 15 nights the engine reads. full: each program's own "
                           "window, roughly a semester (long).")

    run = sub.add_parser("run", help="Measure realtime plans and write the report.")
    _add_clone_args(run)
    run.add_argument("--source", choices=("clones", "odb"), default="clones",
                     help="clones (default): the live ODB, fetched once at the start and cloned "
                          "to --sizes per site. odb: no clones; every plan fetches from the ODB "
                          "itself, as realtime does, and --sizes does not apply.")
    run.add_argument("--visibility", choices=("local", "sight"), required=True)
    run.add_argument("--label", required=True, help="Report section name, e.g. local, sight-15n, sight-full.")
    run.add_argument("--vis-end", metavar="semester|realtime-default|YYYY-MM-DD", default="semester",
                     help="How far the engine reads visibility, set as BuildParameters.visibility_end "
                          "the way an operator would. 'semester' (default): to the end of the "
                          "night's semester, which remaining visibility needs. 'realtime-default': "
                          "the 15 nights of default_operation_parameters. Or a UT date (08:00 "
                          "anchor, inclusive).")
    run.add_argument("--plans", type=int, default=3, help="Measured plans per size (default 3).")
    run.add_argument("--offset-hours", type=float, default=3.0,
                     help="Plan i replans at earliest evening twilight + i × this (default 3).")
    run.add_argument("--iq", type=float, default=0.2, help="Image quality for both sites (default 0.2).")
    run.add_argument("--cc", type=float, default=0.5, help="Cloud cover for both sites (default 0.5).")
    run.add_argument("--no-warmup", action="store_true", help="Skip the unmeasured warm-up plan.")
    run.add_argument("--log-level", default="WARNING", choices=("DEBUG", "INFO", "WARNING", "ERROR"),
                     help="Scheduler log level during the run (default WARNING).")

    args = parser.parse_args()

    if args.command == "seed-sight":
        _require_database_url("seed-sight")
        _prepare_env("sight")
        return asyncio.run(_seed_sight(args))

    if not _LABEL_ARG.match(args.label):
        sys.exit("--label may only hold letters, digits, '.', '_' and '-'.")
    if args.plans < 1:
        sys.exit("--plans must be at least 1.")
    if args.source == "odb" and args.sizes:
        sys.exit("--sizes only applies to --source clones; live runs use whatever the ODB holds.")
    if args.visibility == "sight":
        _require_database_url("run --visibility sight")
    _prepare_env(args.visibility)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
