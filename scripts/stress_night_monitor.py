#!/usr/bin/env python3
"""Night monitor stress test: mock events through the real night monitor and the real engine.

Measures, for every event it injects:

1. response time: the event reaches the listener -> a plan is delivered to subscribers;
2. loss: whether each event is handled, and where the ones that are not disappear;
3. processing time: the event reaches the listener -> a schedule event is sent
   (SchedulerQueue.add_schedule_event).

Mock subscription sources (ODB obscalc, Weather, Resource) and a mock night clock feed the
real NightMonitor (EventListener -> EventConsumer -> handlers), the real SchedulerQueue and
the real EngineRT, wired by a real SchedulerProcess. Nothing in the backend is changed: the
sources, the night clock and the timing hooks are monkeypatched from here and undone at the
end, the same way scripts/stress_realtime.py does it.

The engine is real, so a run needs what stress_realtime needs: program data (clones of the
programs active on the night, fetched once at the start, or --source odb) through the
backend's own GPP client and whatever environment it is configured with. This script never
reads credentials itself.

Usage (from the repo root):

    # Offline: patch targets exist, every mock payload validates, the monitor half of the
    # pipeline handles mock events. No network, no files written.
    .venv/bin/python scripts/stress_night_monitor.py selfcheck

    # The case catalog, one case at a time, each outcome checked against what it should be.
    .venv/bin/python scripts/stress_night_monitor.py cases --list
    .venv/bin/python scripts/stress_night_monitor.py cases --visibility local --size 25 \\
        --label nm-smoke --cases weather_valid_gn
    .venv/bin/python scripts/stress_night_monitor.py cases --visibility local --size 250 --label nm-cases

    # A sustained mixed stream: backlog growth, latency under load, loss on reconnects.
    .venv/bin/python scripts/stress_night_monitor.py load --visibility local --size 250 \\
        --label nm-load --rate 0.2 --duration 600 --drop-every 50

Results land in stress/results/night_monitor_report.md (one section per --label, comparison
on top) and stress/results/night_monitor_<label>.jsonl (one record per injected event and per
schedule event). Add OTEL_EXPORTER_OTLP_ENDPOINT to push the stage timings to Grafana as
operation="stress.nm.<stage>", tagged deployment.environment=stress.

--visibility sight needs DATABASE_URL and a Sight DB seeded by stress_realtime.py seed-sight
with the same --night, --cap, --seed and --programs. Warning: in sight mode the READY cases
call the live ODB REST visibility-changes endpoint from inside the consumer, and the
new-observation cases write Target and visibility rows (STRESS-NM-* targets). Only point it at
a local docker DB.

Known backend bugs the catalog reproduces (reported as KNOWN BUG, not FAIL):
- CREATED + READY raises AttributeError on target_environment.first_science_target
  (odb_event_handler.py _on_created_edit): the generated payload model has no such method.
- HARD_DELETE of a planned observation never replans: a str label is compared with a set of
  lucupy ObservationID (odb_event_handler.py _on_deleted_edit).
"""

import argparse
import asyncio
import contextvars
import copy
import functools
import itertools
import json
import logging
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stress_realtime as rt  # noqa: E402  (also puts backend/ on sys.path)

RESULTS_DIR = rt.STRESS_DIR / "results"
REPORT_PATH = RESULTS_DIR / "night_monitor_report.md"
PROCESS_ID = "stress-nm"
SITE_NAMES = ("GN", "GS")
INSTRUMENT_FOR_SITE = {"GN": "GMOS_NORTH", "GS": "GMOS_SOUTH"}
# Origins of a schedule event that the injected event caused directly, as opposed to a timer
# it armed.
DIRECT_ORIGINS = ("handler", "night_tracker", "on_demand")
TIMER_ORIGINS = ("idle_timer", "execution_timer")
# Index of each phenomenon in NightTracker.calculate_night_events.
NIGHT_INDEX = {
    "midnight": 0,
    "sunset": 1,
    "sunrise": 2,
    "evening": 3,
    "morning": 4,
    "moonrise": 5,
    "moonset": 6,
}
DEFAULT_MIX = (
    "weather=0.4,odb_ready_in_plan=0.3,odb_ongoing_out_of_order=0.1,"
    "odb_completed_not_in_plan=0.2"
)

# Which injected event the running code is working for, and on whose behalf. asyncio copies
# both into every task it creates, so a timer armed while handling an event still knows the
# event that armed it.
_CAUSE: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "stress_nm_cause", default=None
)
_ORIGIN: contextvars.ContextVar[str] = contextvars.ContextVar(
    "stress_nm_origin", default="unknown"
)


# --------------------------------------------------------------------------------------
# Ledger: every timestamp the hooks record
# --------------------------------------------------------------------------------------


@dataclass
class Injection:
    """One event the harness sent. Times are seconds since the ledger started."""

    inj_id: str
    case: str
    source: str  # odb | weather | resource | night | on_demand
    kind: str
    t_sent: float
    expect: Optional[str] = None  # load mode: 'plan' or 'none'
    t_received: Optional[float] = None  # the subscription handed it to the listener
    t_handle_start: Optional[float] = None
    t_handle_end: Optional[float] = None
    error: Optional[str] = None  # the handler raised; the consumer skipped it
    lost: Optional[str] = None  # disconnected | in_flight | validation
    filtered: bool = False  # a night event the tracker does not enqueue
    sched_ids: List[str] = field(default_factory=list)


@dataclass
class SchedRecord:
    """One schedule event put on the SchedulerQueue, and what became of it."""

    sched_id: str
    event_type: str
    site: str
    description: str
    origin: str
    cause: Optional[str]
    t_sched: float
    t_plan_start: Optional[float] = None
    t_plan_end: Optional[float] = None
    plan_error: Optional[str] = None
    end_of_night: bool = False
    plan_index: Optional[int] = None
    t_delivered: Optional[float] = None
    delivered_error: bool = False


class Ledger:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.injections: Dict[str, Injection] = {}
        self.scheds: Dict[str, SchedRecord] = {}
        # id(payload) -> (payload, inj_id). The payload is kept alive so its id is never reused.
        self.payloads: Dict[int, Tuple[Any, str]] = {}
        # Plans computed and not yet seen by the subscriber, in the order the engine made them.
        self.computed: List[dict] = []
        self.handling = 0
        self.planning = 0
        self.plan_count = 0
        self.unmatched_deliveries: List[str] = []
        self.timer_arms: List[dict] = []
        self.shorten: Dict[str, Dict[str, int]] = {}
        self.shortened_tasks: List[asyncio.Task] = []
        self.timer_wait = 10.0
        self.samples: List[Tuple[float, int, int]] = []
        self.max_event_queue = 0
        self.max_sched_queue = 0
        self.payload_copy_s: List[float] = []
        self._seq = itertools.count(1)

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def new_injection(self, case: str, source: str, kind: str) -> Injection:
        inj = Injection(f"i{next(self._seq):05d}", case, source, kind, self.now())
        self.injections[inj.inj_id] = inj
        return inj

    def register_payload(self, payload: Any, inj_id: str) -> None:
        self.payloads[id(payload)] = (payload, inj_id)

    def lookup_payload(self, payload: Any) -> Optional[str]:
        entry = self.payloads.get(id(payload))
        return entry[1] if entry is not None and entry[0] is payload else None

    def case_injections(self, case: str) -> List[Injection]:
        return [i for i in self.injections.values() if i.case == case]


# --------------------------------------------------------------------------------------
# Payloads
# --------------------------------------------------------------------------------------

LOOSE = {
    "imageQuality": "TWO_POINT_ZERO",
    "cloudExtinction": "THREE_POINT_ZERO",
    "skyBackground": "BRIGHT",
    "waterVapor": "WET",
    "elevationRange": {"airMass": {"min": 1.0, "max": 2.0}, "hourAngle": None},
}
TIGHT = {
    "imageQuality": "POINT_ONE",
    "cloudExtinction": "ZERO",
    "skyBackground": "DARKEST",
    "waterVapor": "VERY_DRY",
    "elevationRange": {"airMass": {"min": 1.0, "max": 1.2}, "hourAngle": None},
}
_DEFAULT = object()
_internal_ids = itertools.count(0x5000)
_synthetic_labels = itertools.count(1)


def synthetic_label() -> str:
    """An observation label no real program uses, so it is never in a plan."""
    return f"G-2099A-9999-Q-{next(_synthetic_labels):04d}"


def odb_payload(
    label: Optional[str],
    *,
    site: str = "GN",
    state: str = "READY",
    edit_type: str = "UPDATED",
    constraints: Optional[dict] = None,
    active: Tuple[str, str] = ("2026-08-01", "2027-01-31"),
    instrument: Any = _DEFAULT,
    value: bool = True,
    reference: bool = True,
    workflow: bool = True,
) -> dict:
    """An obscalcUpdate frame as the ODB subscription sends it, before model validation.

    Every Optional field without a default is present (pydantic requires it). The target
    carries a STRESS-NM-* name, so anything sight mode stores for it is easy to find.
    """
    body = {
        "id": f"o-{next(_internal_ids):x}",
        "reference": {"label": label} if reference else None,
        "observationTime": None,
        "program": {"active": {"start": active[0], "end": active[1]}},
        "workflow": {"value": {"state": state}} if workflow else None,
        "execution": {"visits": {"matches": []}},
        "targetEnvironment": {
            "asterism": [
                {
                    "name": f"STRESS-NM-{label}",
                    "sidereal": {
                        "ra": {
                            "hours": 10.0,
                            "hms": "10:00:00.000000",
                            "degrees": 150.0,
                        },
                        "dec": {"degrees": 20.0, "dms": "+20:00:00.000000"},
                        "epoch": "J2000.000",
                    },
                    "nonsidereal": None,
                }
            ],
            "explicitBase": None,
        },
        "constraintSet": copy.deepcopy(constraints or LOOSE),
        "schedulingConstraints": {"timingWindows": []},
        "instrument": INSTRUMENT_FOR_SITE[site]
        if instrument is _DEFAULT
        else instrument,
    }
    return {
        "obscalcUpdate": {
            "oldCalculationState": "CALCULATING",
            "newCalculationState": "READY",
            "editType": edit_type,
            "value": body if value else None,
        }
    }


def weather_payload(site: str, iq: Any, cc: Any) -> dict:
    return {
        "weatherUpdates": {
            "site": site,
            "imageQuality": iq,
            "cloudCover": cc,
            "windDirection": 0.0,
            "windSpeed": 0.0,
        }
    }


def resource_payload(status: str, name: str = "GMOS-N") -> dict:
    return {"resource_status": status, "resource_name": name}


# --------------------------------------------------------------------------------------
# Mock sources: each one models a server feed
# --------------------------------------------------------------------------------------


class _Fault:
    """end: the server completes the stream. closed: the socket drops. fatal: not retried."""

    def __init__(self, kind: str) -> None:
        self.kind = kind


class MockFeed:
    """What a subscriber of one subscription receives while it is connected.

    emit() is the server sending: when nobody is subscribed the event is gone, as it is for
    a real subscription that is reconnecting.
    """

    def __init__(self, name: str, ledger: Ledger) -> None:
        self.name = name
        self.ledger = ledger
        self._queue: Optional[asyncio.Queue] = None
        self.connections = 0
        self.dead = False
        self.downtimes: List[Tuple[float, Optional[float]]] = []

    @property
    def connected(self) -> bool:
        return self._queue is not None

    def pending(self) -> int:
        return 0 if self._queue is None else self._queue.qsize()

    def emit(self, item: Any, inj: Optional[Injection]) -> bool:
        if self._queue is None:
            if inj is not None:
                inj.lost = "disconnected"
            return False
        self._queue.put_nowait((item, inj))
        return True

    def fault(self, kind: str) -> bool:
        """False when there is no connection to break."""
        if self._queue is None:
            return False
        self._queue.put_nowait((_Fault(kind), None))
        return True

    def _disconnect(self, queue: asyncio.Queue) -> None:
        if self._queue is not queue:
            return
        self._queue = None
        self.downtimes.append((self.ledger.now(), None))
        # Whatever was still on the wire went down with the socket.
        while not queue.empty():
            item, inj = queue.get_nowait()
            if inj is not None and inj.t_received is None:
                inj.lost = "in_flight"

    async def stream(self, convert: Callable[[Any], Any]):
        queue: asyncio.Queue = asyncio.Queue()
        self._queue = queue
        self.connections += 1
        if self.downtimes and self.downtimes[-1][1] is None:
            self.downtimes[-1] = (self.downtimes[-1][0], self.ledger.now())
        try:
            while True:
                item, inj = await queue.get()
                if isinstance(item, _Fault):
                    self._disconnect(queue)
                    if item.kind == "end":
                        return
                    if item.kind == "closed":
                        from websockets import ConnectionClosedError

                        raise ConnectionClosedError(None, None)
                    self.dead = True
                    raise RuntimeError(
                        f"stress: mock subscription '{self.name}' failed for good"
                    )
                try:
                    payload = convert(item)
                except Exception:
                    # The real client validates inside its generator, so a bad frame tears the
                    # subscription down rather than reaching a handler.
                    if inj is not None:
                        inj.lost = "validation"
                    self._disconnect(queue)
                    raise
                if inj is not None:
                    inj.t_received = self.ledger.now()
                    self.ledger.register_payload(payload, inj.inj_id)
                yield payload
        finally:
            self._disconnect(queue)


class _AsyncSession:
    """The weather path opens `async with client as session` before subscribing."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> bool:
        return False


class MockSources:
    """The three feeds, the weather state the engine reads, and the night clock queue."""

    def __init__(
        self,
        B: SimpleNamespace,
        ledger: Ledger,
        iq: float,
        cc: float,
        resource_subscription: bool,
    ) -> None:
        self.B = B
        self.ledger = ledger
        self.odb = MockFeed("observation_edit", ledger)
        self.weather = MockFeed("weather_change", ledger)
        self.resource = MockFeed("resource_edit", ledger)
        self.resource_subscription = resource_subscription
        self.weather_state = {
            site: {
                "site": site,
                "imageQuality": iq,
                "cloudCover": cc,
                "windDirection": 0.0,
                "windSpeed": 0.0,
            }
            for site in SITE_NAMES
        }
        self.weather_reads = 0
        self.night_queue: Optional[asyncio.Queue] = None

    def feeds(self) -> List[MockFeed]:
        return [self.odb, self.weather] + (
            [self.resource] if self.resource_subscription else []
        )

    def apply_weather(self, item: dict) -> dict:
        """The weather service updates its state before it pushes the update."""
        try:
            update = item["weatherUpdates"]
            site = self.B.Site[update["site"]].name
            self.B.ImageQuality(update["imageQuality"])
            self.B.CloudCover(update["cloudCover"])
        except Exception:
            return item
        self.weather_state[site] = dict(update)
        return item

    def source_classes(self) -> Tuple[type, type, type]:
        """Stand-ins for the three classes EventListener builds its sources from."""
        mocks = self
        EventSourceType = self.B.EventSourceType
        validate = self.B.SchedulerObservationsUpdates.model_validate

        class MockResourceEventSource:
            RESOURCE_EDIT = "resource_edit"

            def __init__(self, client):
                self.source_type = EventSourceType.RESOURCE

            def get_current_state(self):
                return None

            def subscriptions(self):
                # Production has no resource subscription; the resource cases add one.
                if not mocks.resource_subscription:
                    return []
                return [
                    (
                        "resource_edit",
                        lambda _client: mocks.resource.stream(lambda x: x),
                        None,
                    )
                ]

        class MockWeatherEventSource:
            WEATHER_CHANGE = "weather_change"

            def __init__(self, client):
                self.source_type = EventSourceType.WEATHER

            async def get_current_state(self):
                mocks.weather_reads += 1
                return copy.deepcopy(list(mocks.weather_state.values()))

            def subscriptions(self):
                return [
                    (
                        "weather_change",
                        lambda _session: mocks.weather.stream(mocks.apply_weather),
                        _AsyncSession(),
                    )
                ]

        class MockODBEventSource:
            OBSERVATION_EDIT = "observation_edit"
            VISIT_EXECUTED = "visit_executed"

            def __init__(self, client):
                self.source_type = EventSourceType.ODB

            def subscriptions(self):
                return [
                    (
                        "observation_edit",
                        lambda _client: mocks.odb.stream(validate),
                        None,
                    )
                ]

        return MockResourceEventSource, MockWeatherEventSource, MockODBEventSource

    async def track(self, tracker) -> None:
        """Replaces NightTracker.start_tracking: night events arrive when a case sends them.

        The real tracker ignores its date and sleeps until tonight's real twilights. What is
        under test is everything after it, so the harness decides when, and keeps the real
        filter on which night events become plans.
        """
        self.night_queue = asyncio.Queue()
        while True:
            event, inj = await self.night_queue.get()
            inj.t_received = inj.t_handle_start = self.ledger.now()
            if self.B.NightTracker._should_trigger_plan(event):
                cause, origin = _CAUSE.set(inj.inj_id), _ORIGIN.set("night_tracker")
                try:
                    await tracker.scheduler_queue.add_schedule_event(event)
                finally:
                    _CAUSE.reset(cause)
                    _ORIGIN.reset(origin)
            else:
                inj.filtered = True
            inj.t_handle_end = self.ledger.now()
            self.night_queue.task_done()


# --------------------------------------------------------------------------------------
# Backend imports and hooks
# --------------------------------------------------------------------------------------


def _backend() -> SimpleNamespace:
    """Everything the harness touches, imported after _prepare_env pinned the mode."""
    # backend/definitions.py reads a lone command-line argument as a log level.
    argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        return _import_backend()
    finally:
        sys.argv = argv


# name -> (module, attribute); no attribute means the module itself. Order matters: the engine
# package sits on an import cycle with the night monitor that only resolves when the engine is
# imported first, as the app and stress_realtime do.
_BACKEND = (
    ("EngineRT", "scheduler.engine.engineRT", "EngineRT"),
    ("ALL_SITES", "lucupy.minimodel", "ALL_SITES"),
    ("CloudCover", "lucupy.minimodel", "CloudCover"),
    ("ImageQuality", "lucupy.minimodel", "ImageQuality"),
    ("NightIndex", "lucupy.minimodel", "NightIndex"),
    ("ObservationStatus", "lucupy.minimodel", "ObservationStatus"),
    ("Site", "lucupy.minimodel", "Site"),
    (
        "SchedulerObservationsUpdates",
        "gpp_client.generated.scheduler_observations_updates",
        "SchedulerObservationsUpdates",
    ),
    ("simulationbuilder", "scheduler.core.builder.simulationbuilder", None),
    ("event_listener", "scheduler.night_monitor.event_listener", None),
    ("schedule_id_var", "scheduler.context", "schedule_id_var"),
    ("SchedulerModes", "scheduler.core.builder.modes", "SchedulerModes"),
    ("NightlyTimelineStore", "scheduler.core.events.queue", "NightlyTimelineStore"),
    ("EndOfNightEvent", "scheduler.core.events.queue.events", "EndOfNightEvent"),
    (
        "OnDemandScheduleEvent",
        "scheduler.core.events.queue.events",
        "OnDemandScheduleEvent",
    ),
    (
        "SchedulerQueue",
        "scheduler.core.events.queue.scheduler_queue_client",
        "SchedulerQueue",
    ),
    ("BuildParameters", "scheduler.engine.params", "BuildParameters"),
    ("SchedulerParameters", "scheduler.engine.params", "SchedulerParameters"),
    ("build_params_store", "scheduler.engine.params", "build_params_store"),
    ("NightPlansError", "scheduler.graphql_mid.types", "NightPlansError"),
    ("EventConsumer", "scheduler.night_monitor.event_consumer", "EventConsumer"),
    (
        "EventHandler",
        "scheduler.night_monitor.event_handlers.event_handler",
        "EventHandler",
    ),
    (
        "build_conditions",
        "scheduler.night_monitor.event_handlers.obscalc_visibility",
        "build_conditions",
    ),
    (
        "site_key_from_instrument",
        "scheduler.night_monitor.event_handlers.obscalc_visibility",
        "site_key_from_instrument",
    ),
    (
        "ODBEventHandler",
        "scheduler.night_monitor.event_handlers.odb_event_handler",
        "ODBEventHandler",
    ),
    (
        "SchedulerIdleTimer",
        "scheduler.night_monitor.event_handlers.odb_event_handler",
        "SchedulerIdleTimer",
    ),
    (
        "ResourceEventHandler",
        "scheduler.night_monitor.event_handlers.resource_event_handler",
        "ResourceEventHandler",
    ),
    (
        "WeatherEventHandler",
        "scheduler.night_monitor.event_handlers.weather_event_handler",
        "WeatherEventHandler",
    ),
    ("EventListener", "scheduler.night_monitor.event_listener", "EventListener"),
    ("EventSourceType", "scheduler.night_monitor.event_sources", "EventSourceType"),
    ("NightMonitor", "scheduler.night_monitor.night_monitor", "NightMonitor"),
    ("NightTracker", "scheduler.night_monitor.night_tracker", "NightTracker"),
    (
        "SchedulerProcess",
        "scheduler.orchestration.scheduler_process",
        "SchedulerProcess",
    ),
    (
        "plan_response_subscribers",
        "scheduler.shared_queue",
        "plan_response_subscribers",
    ),
)


def _import_backend() -> SimpleNamespace:
    import importlib

    backend = SimpleNamespace()
    for name, module, attribute in _BACKEND:
        imported = importlib.import_module(module)
        setattr(
            backend,
            name,
            imported if attribute is None else getattr(imported, attribute),
        )
    return backend


def _install_hooks(
    patches: "rt._Patches",
    B: SimpleNamespace,
    ledger: Ledger,
    mocks: MockSources,
    timer_mode: str,
) -> None:
    """Every patch goes in before SchedulerProcess.run: the listener picks its sources and the
    handlers bind their dispatch maps when they are constructed."""
    resource_cls, weather_cls, odb_cls = mocks.source_classes()
    patches.set(B.event_listener, "ResourceEventSource", resource_cls)
    patches.set(B.event_listener, "WeatherEventSource", weather_cls)
    patches.set(B.event_listener, "ODBEventSource", odb_cls)

    def tracker(_original):
        async def start_tracking(self):
            await mocks.track(self)

        return start_tracking

    patches.wrap(B.NightTracker, "start_tracking", tracker)

    if timer_mode == "quiet":
        # Read when the idle timer is armed. At 10 min, every quiet stretch adds a replan that
        # no case asked for; the timer cases shorten their own timer instead.
        patches.set(B.ODBEventHandler, "WAITING_THRESHOLD", timedelta(hours=6))

    # H2: the handler, the single entry every subscription event goes through.
    def handle(original):
        @functools.wraps(original)
        async def wrapper(self, sub_name, raw_event):
            inj_id = ledger.lookup_payload(raw_event)
            inj = ledger.injections.get(inj_id) if inj_id else None
            cause, origin = _CAUSE.set(inj_id), _ORIGIN.set("handler")
            ledger.handling += 1
            if inj is not None:
                inj.t_handle_start = ledger.now()
            try:
                return await original(self, sub_name, raw_event)
            except Exception as exc:
                if inj is not None:
                    inj.error = f"{type(exc).__name__}: {rt._short(exc)}"
                raise
            finally:
                if inj is not None:
                    inj.t_handle_end = ledger.now()
                ledger.handling -= 1
                _CAUSE.reset(cause)
                _ORIGIN.reset(origin)

        return wrapper

    patches.wrap(B.EventHandler, "handle", handle)

    # H3: a schedule event is sent.
    def add_schedule_event(original):
        @functools.wraps(original)
        async def wrapper(self, event):
            if event is not None:
                cause = _CAUSE.get()
                site = event.site
                site_name = (
                    "ALL"
                    if isinstance(site, (set, frozenset))
                    else getattr(site, "name", str(site))
                )
                record = SchedRecord(
                    str(event.id),
                    type(event).__name__,
                    site_name,
                    str(event.description),
                    _ORIGIN.get(),
                    cause,
                    ledger.now(),
                )
                ledger.scheds[record.sched_id] = record
                if cause in ledger.injections:
                    ledger.injections[cause].sched_ids.append(record.sched_id)
            return await original(self, event)

        return wrapper

    patches.wrap(B.SchedulerQueue, "add_schedule_event", add_schedule_event)

    # H4: the engine plans. Must stay a coroutine function: consume_events checks.
    def compute_event_plan(original):
        @functools.wraps(original)
        async def wrapper(self, event):
            sched_id = str(event.id)
            record = ledger.scheds.get(sched_id)
            cause = _CAUSE.set(record.cause if record else None)
            origin = _ORIGIN.set("engine")
            # Keys the engine's own perf events (engine.plan, scp.*) by schedule event.
            run_id = B.schedule_id_var.set(sched_id)
            ledger.planning += 1
            ledger.plan_count += 1
            entry = {
                "sched_id": sched_id,
                "result": None,
                "ok": False,
                "delivered": False,
                "eon": isinstance(event, B.EndOfNightEvent),
            }
            if record is not None:
                record.t_plan_start = ledger.now()
                record.plan_index = ledger.plan_count
            try:
                entry["result"] = await original(self, event)
                entry["ok"] = True
                return entry["result"]
            except BaseException as exc:
                if record is not None:
                    record.plan_error = f"{type(exc).__name__}: {rt._short(exc)}"
                raise
            finally:
                if record is not None:
                    record.t_plan_end = ledger.now()
                    record.end_of_night = entry["eon"]
                ledger.planning -= 1
                # A finished end-of-night plan is never published; a failed one is, as an error.
                if not (entry["eon"] and entry["ok"]):
                    ledger.computed.append(entry)
                B.schedule_id_var.reset(run_id)
                _ORIGIN.reset(origin)
                _CAUSE.reset(cause)

        return wrapper

    patches.wrap(B.EngineRT, "compute_event_plan", compute_event_plan)

    def origin_tag(tag: str):
        def make(original):
            @functools.wraps(original)
            async def wrapper(*args, **kwargs):
                token = _ORIGIN.set(tag)
                try:
                    return await original(*args, **kwargs)
                finally:
                    _ORIGIN.reset(token)

            return wrapper

        return make

    patches.wrap(B.ODBEventHandler, "_on_idle_timeout", origin_tag("idle_timer"))
    patches.wrap(
        B.ODBEventHandler, "_on_execution_timeout", origin_tag("execution_timer")
    )
    patches.wrap(B.ODBEventHandler, "on_plan_published", origin_tag("plan_published"))

    # Timers: record every arm, and shorten the one a timer case asked for, once. A flag in
    # the context would be inherited by every re-arm and replan forever.
    def timer_set(original):
        @functools.wraps(original)
        def wrapper(self, delay, callback, with_offset=False):
            name = getattr(getattr(callback, "func", None), "__name__", "")
            kind = (
                "idle"
                if "idle" in name
                else "execution"
                if "execution" in name
                else name
            )
            args = getattr(callback, "args", ())
            site = getattr(args[0], "name", str(args[0])) if args else None
            cause, origin = _CAUSE.get(), _ORIGIN.get()
            shortened = False
            budget = ledger.shorten.get(cause)
            if budget and budget.get(kind, 0) > 0 and origin == "handler":
                budget[kind] -= 1
                delay, with_offset, shortened = ledger.timer_wait, False, True
            ledger.timer_arms.append(
                {
                    "t": ledger.now(),
                    "kind": kind,
                    "site": site,
                    "cause": cause,
                    "origin": origin,
                    "delay_s": delay,
                    "shortened": shortened,
                }
            )
            result = original(self, delay, callback, with_offset)
            if shortened and getattr(self, "_task", None) is not None:
                ledger.shortened_tasks.append(self._task)
            return result

        return wrapper

    patches.wrap(B.SchedulerIdleTimer, "set", timer_set)


async def _collect_deliveries(
    queue: asyncio.Queue, B: SimpleNamespace, ledger: Ledger
) -> None:
    """H5: the harness is the subscriber. The engine publishes in the order it plans."""
    while True:
        obj = await queue.get()
        t = ledger.now()
        if isinstance(obj, B.NightPlansError):
            entry = next(
                (e for e in ledger.computed if not e["ok"] and not e["delivered"]), None
            )
        else:
            entry = next(
                (
                    e
                    for e in ledger.computed
                    if e["result"] is obj and not e["delivered"]
                ),
                None,
            )
        if entry is None:
            # An error with no failed plan behind it is the engine task itself dying.
            ledger.unmatched_deliveries.append(
                f"{type(obj).__name__}: {getattr(obj, 'error', '')}"[:200]
            )
            continue
        entry["delivered"] = True
        record = ledger.scheds.get(entry["sched_id"])
        if record is not None:
            record.t_delivered = t
            record.delivered_error = isinstance(obj, B.NightPlansError)


async def _sample(ledger: Ledger, event_queue: asyncio.Queue, sched_queue: Any) -> None:
    while True:
        eq, sq = event_queue.qsize(), sched_queue._queue.qsize()
        ledger.samples.append((round(ledger.now(), 2), eq, sq))
        ledger.max_event_queue = max(ledger.max_event_queue, eq)
        ledger.max_sched_queue = max(ledger.max_sched_queue, sq)
        await asyncio.sleep(0.5)


# --------------------------------------------------------------------------------------
# Context the cases run in
# --------------------------------------------------------------------------------------


class Ctx:
    def __init__(
        self,
        args,
        B: SimpleNamespace,
        ledger: Ledger,
        mocks: MockSources,
        proc,
        night_events: Dict[str, list],
        active: Tuple[str, str],
    ) -> None:
        self.args = args
        self.B = B
        self.ledger = ledger
        self.mocks = mocks
        self.proc = proc
        self.night_events = night_events
        self.active = active
        self.case = "setup"
        self.notes: List[str] = []

    # --- plumbing --------------------------------------------------------------------

    @property
    def store(self):
        return self.proc.nightly_timeline_store

    @property
    def odb_handler(self):
        return self.proc.night_monitor.consumer.odb_handler

    @property
    def engine_alive(self) -> bool:
        task = self.proc._engine_task
        return task is not None and not task.done()

    def site(self, name: str):
        return self.B.Site[name]

    def note(self, text: str) -> None:
        self.notes.append(text)

    # --- sending ---------------------------------------------------------------------

    def _new(
        self, source: str, kind: str, shorten: Optional[Dict[str, int]] = None
    ) -> Injection:
        inj = self.ledger.new_injection(self.case, source, kind)
        if shorten:
            self.ledger.shorten[inj.inj_id] = dict(shorten)
        return inj

    def odb(
        self, payload: dict, kind: str, shorten: Optional[Dict[str, int]] = None
    ) -> Injection:
        inj = self._new("odb", kind, shorten)
        self.mocks.odb.emit(payload, inj)
        return inj

    def weather(self, payload: dict, kind: str) -> Injection:
        inj = self._new("weather", kind)
        self.mocks.weather.emit(payload, inj)
        return inj

    def resource(self, payload: dict, kind: str) -> Injection:
        inj = self._new("resource", kind)
        self.mocks.resource.emit(payload, inj)
        return inj

    def night(self, event, kind: str) -> Injection:
        inj = self._new("night", kind)
        self.mocks.night_queue.put_nowait((event, inj))
        return inj

    async def on_demand(self, event, kind: str) -> Injection:
        """As the schedule_v2 / on_demand_schedule mutations do: straight onto the queue."""
        inj = self._new("on_demand", kind)
        inj.t_received = inj.t_handle_start = self.ledger.now()
        cause, origin = _CAUSE.set(inj.inj_id), _ORIGIN.set("on_demand")
        try:
            await self.proc.scheduler_queue.add_schedule_event(event)
        finally:
            _CAUSE.reset(cause)
            _ORIGIN.reset(origin)
        inj.t_handle_end = self.ledger.now()
        return inj

    def night_event(self, site: str, name: str):
        """A fresh copy (new id) of one of the simulated night's events."""
        return replace(self.night_events[site][NIGHT_INDEX[name]])

    def end_of_night_event(self):
        """Built exactly as NightTracker._compute_sorted_night_events builds it."""
        last = max(e.time for events in self.night_events.values() for e in events)
        return self.B.EndOfNightEvent(
            description="End of Night", time=last + timedelta(minutes=5), site="Both"
        )

    # --- reading the plan in effect --------------------------------------------------

    async def plan(self, site: str):
        return await self.store.last_plan(self.site(site))

    async def when(self, site: Optional[str] = None) -> datetime:
        """The handlers' own sense of now (simulated night, advancing in real time)."""
        return await self.odb_handler._reference_time(self.site(site) if site else None)

    @staticmethod
    def visit_end(plan, visit) -> datetime:
        return visit.start_time + visit.time_slots * plan.time_slot_length

    async def plans(self) -> Dict[str, Any]:
        out = {}
        for name in SITE_NAMES:
            plan = await self.plan(name)
            if plan is not None and plan.visits:
                out[name] = plan
        return out

    async def covering_visit(
        self, *, ready: bool = True, margin_s: float = 120.0, wait_s: float = 90.0
    ) -> Optional[Tuple[str, Any, Any]]:
        """A visit the plan wants running now, with time to spare before it ends.

        The reference time advances in real time, so a visit about to start is waited for.
        """
        for attempt in range(2):
            soonest = None
            for name, plan in (await self.plans()).items():
                when = await self.when(name)
                for visit in plan.visits:
                    if (
                        ready
                        and visit.observation.status
                        is not self.B.ObservationStatus.READY
                    ):
                        continue
                    end = self.visit_end(plan, visit)
                    if (
                        visit.start_time <= when
                        and (end - when).total_seconds() >= margin_s
                    ):
                        return name, plan, visit
                    gap = (visit.start_time - when).total_seconds()
                    if (
                        0 < gap <= wait_s
                        and (end - visit.start_time).total_seconds() >= margin_s + gap
                    ):
                        soonest = gap if soonest is None else min(soonest, gap)
            if soonest is None or attempt == 1:
                return None
            await asyncio.sleep(soonest + 1.0)
        return None

    async def idle_visit(
        self,
        *,
        ready: bool = False,
        exclude: Tuple[str, ...] = (),
        site: Optional[str] = None,
    ) -> Optional[Tuple[str, Any, Any]]:
        """A planned visit the plan does not want running now."""
        for name, plan in (await self.plans()).items():
            if site is not None and name != site:
                continue
            when = await self.when(name)
            for visit in reversed(plan.visits):
                if visit.observation.id.id in exclude:
                    continue
                if (
                    ready
                    and visit.observation.status is not self.B.ObservationStatus.READY
                ):
                    continue
                if not (visit.start_time <= when < self.visit_end(plan, visit)):
                    return name, plan, visit
        return None

    async def ready_visit(self) -> Optional[Tuple[str, Any, Any]]:
        """Any planned visit whose observation is READY in the plan."""
        for name, plan in (await self.plans()).items():
            for visit in plan.visits:
                if visit.observation.status is self.B.ObservationStatus.READY:
                    return name, plan, visit
        return None

    async def unplanned_label(self, site: str, plan) -> str:
        """An observation visible tonight at the site that the plan left out.

        Synthetic under sight visibility: a new-observation event stores visibility computed
        from the mock target under that label, which would overwrite a real clone's rows.
        """
        if self.args.visibility == "sight":
            return synthetic_label()
        engine = self.proc.engine
        scp = getattr(engine, "scp", None)
        if scp is not None:
            collector = scp.collector
            planned = {v.observation.id for v in plan.visits}
            target = self.site(site)
            for oid in sorted(
                collector.get_visible_observations_for_night(self.B.NightIndex(0)),
                key=lambda o: o.id,
            ):
                obs = collector.get_observation(oid)
                if obs is not None and obs.site == target and oid not in planned:
                    return oid.id
        return synthetic_label()

    # --- waiting ---------------------------------------------------------------------

    async def wait_for(self, predicate: Callable[[], bool], timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                return False
            await asyncio.sleep(0.05)
        return True

    async def wait_connected(
        self, feeds: List[MockFeed], timeout: float = 30.0
    ) -> bool:
        return await self.wait_for(lambda: all(f.connected for f in feeds), timeout)

    async def wait_reconnected(
        self, feed: MockFeed, connections_before: int, timeout: float = 45.0
    ) -> bool:
        return await self.wait_for(
            lambda: feed.connected and feed.connections > connections_before, timeout
        )

    def busy(self) -> List[str]:
        """Why the pipeline is not quiet yet; empty when everything sent has run its course."""
        reasons = []
        if any(f.pending() for f in self.mocks.feeds()):
            reasons.append("mock feed")
        if (
            self.mocks.night_queue is not None
            and self.mocks.night_queue._unfinished_tasks
        ):
            reasons.append("night clock")
        event_queue = self.proc.night_monitor.event_queue
        if getattr(event_queue, "_unfinished_tasks", event_queue.qsize()):
            reasons.append("event queue")
        if self.ledger.handling:
            reasons.append("handler")
        if self.ledger.planning:
            reasons.append("engine")
        if self.engine_alive:
            if self.proc.scheduler_queue._queue.qsize():
                reasons.append("scheduler queue")
            if any(not e["delivered"] for e in self.ledger.computed):
                reasons.append("delivery")
        if any(not t.done() for t in self.ledger.shortened_tasks):
            reasons.append("timer")
        return reasons

    async def wait_settled(
        self, timeout: float, quiet: float = 0.5
    ) -> Tuple[bool, List[str]]:
        deadline = time.monotonic() + timeout
        calm_since = None
        while True:
            reasons = self.busy()
            now = time.monotonic()
            if reasons:
                calm_since = None
            elif calm_since is None:
                calm_since = now
            elif now - calm_since >= quiet:
                return True, []
            if now > deadline:
                return False, reasons
            await asyncio.sleep(0.05)


# --------------------------------------------------------------------------------------
# Case catalog
# --------------------------------------------------------------------------------------


@dataclass
class Expect:
    plans: int = 0  # plans delivered for the events the case sent
    timer: Optional[str] = None  # 'idle' | 'execution': exactly one timer replan
    errors: int = 0  # handler exceptions (the consumer skips the event)
    lost: int = 0  # never reached a handler
    filtered: int = 0  # night events the tracker does not enqueue
    not_published: int = 0  # planned but never published (end of night)
    stranded: int = 0  # schedule events nobody ever planned
    record_only: bool = False  # no assertion, the outcome is recorded


@dataclass
class Case:
    name: str
    group: str
    summary: str
    expect: Expect
    run: Callable[[Ctx], Awaitable[Optional[str]]]
    known_bug: Optional[str] = None
    synthetic: bool = False  # the real subscription would filter it out
    needs_resource: bool = False


async def c_odb_before_plan(ctx: Ctx) -> Optional[str]:
    if await ctx.plans():
        return "a plan is already in effect"
    for name in SITE_NAMES:
        ctx.odb(
            odb_payload(synthetic_label(), site=name, active=ctx.active),
            f"UPDATED READY {name}, no plan yet",
        )
    return None


async def c_night_evening(ctx: Ctx) -> Optional[str]:
    for name in ("GS", "GN"):
        ctx.night(ctx.night_event(name, "evening"), f"evening 12° twilight {name}")
    return None


async def c_night_non_trigger(ctx: Ctx) -> Optional[str]:
    for name in ("sunset", "midnight", "moonrise"):
        ctx.night(ctx.night_event("GN", name), f"{name} GN")
    return None


def _weather_case(site: str, iq: Any, cc: Any, times: int = 1):
    async def run(ctx: Ctx) -> Optional[str]:
        for _ in range(times):
            ctx.weather(
                weather_payload(site, iq, cc), f"weather {site} iq={iq} cc={cc}"
            )
        return None

    return run


async def c_odb_ongoing_expected(ctx: Ctx) -> Optional[str]:
    found = await ctx.covering_visit(ready=True)
    if found is None:
        return "no READY visit covers the reference time"
    name, _plan, visit = found
    ctx.odb(
        odb_payload(
            visit.observation.id.id, site=name, state="ONGOING", active=ctx.active
        ),
        "UPDATED ONGOING, the expected observation",
        shorten={"execution": 1},
    )
    return None


async def c_odb_ongoing_duplicate(ctx: Ctx) -> Optional[str]:
    found = await ctx.covering_visit(ready=True)
    if found is None:
        return "no READY visit covers the reference time"
    name, _plan, visit = found
    for i in range(2):
        ctx.odb(
            odb_payload(
                visit.observation.id.id, site=name, state="ONGOING", active=ctx.active
            ),
            f"UPDATED ONGOING, expected observation #{i + 1}",
            shorten={"execution": 1},
        )
    return None


async def c_odb_ongoing_out_of_order(ctx: Ctx) -> Optional[str]:
    found = await ctx.idle_visit()
    if found is None:
        return "no planned visit outside the current one"
    name, _plan, visit = found
    ctx.odb(
        odb_payload(
            visit.observation.id.id, site=name, state="ONGOING", active=ctx.active
        ),
        "UPDATED ONGOING, planned for later",
    )
    return None


async def c_odb_ongoing_not_in_plan(ctx: Ctx) -> Optional[str]:
    plans = await ctx.plans()
    if not plans:
        return "no plan in effect"
    name, plan = next(iter(plans.items()))
    label = await ctx.unplanned_label(name, plan)
    ctx.odb(
        odb_payload(label, site=name, state="ONGOING", active=ctx.active),
        "UPDATED ONGOING, not in plan",
    )
    return None


def _ready_in_plan(constraints: dict, times: int = 1):
    async def run(ctx: Ctx) -> Optional[str]:
        found = await ctx.ready_visit()
        if found is None:
            return "no READY observation in the plan"
        name, _plan, visit = found
        for i in range(times):
            ctx.odb(
                odb_payload(
                    visit.observation.id.id,
                    site=name,
                    state="READY",
                    constraints=constraints,
                    active=ctx.active,
                ),
                f"UPDATED READY in plan #{i + 1}",
            )
        return None

    return run


async def c_odb_ready_new(ctx: Ctx) -> Optional[str]:
    plans = await ctx.plans()
    if not plans:
        return "no plan in effect"
    name, plan = next(iter(plans.items()))
    label = await ctx.unplanned_label(name, plan)
    ctx.odb(
        odb_payload(label, site=name, state="READY", active=ctx.active),
        "UPDATED READY, not in plan",
    )
    return None


async def c_odb_ready_wrong_site(ctx: Ctx) -> Optional[str]:
    plans = await ctx.plans()
    if len(plans) < 2:
        return "needs a plan at both sites"
    found = await ctx.ready_visit()
    if found is None:
        return "no READY observation in the plan"
    name, _plan, visit = found
    other = "GN" if name == "GS" else "GS"
    ctx.odb(
        odb_payload(
            visit.observation.id.id, site=other, state="READY", active=ctx.active
        ),
        f"UPDATED READY, planned at {name}, instrument says {other}",
    )
    return None


async def c_odb_completed_in_plan(ctx: Ctx) -> Optional[str]:
    found = await ctx.idle_visit()
    if found is None:
        return "no planned visit"
    name, _plan, visit = found
    ctx.odb(
        odb_payload(
            visit.observation.id.id, site=name, state="COMPLETED", active=ctx.active
        ),
        "UPDATED COMPLETED, in plan",
        shorten={"idle": 1},
    )
    return None


async def c_odb_completed_during_wait(ctx: Ctx) -> Optional[str]:
    found = await ctx.covering_visit(ready=True)
    if found is None:
        return "no READY visit covers the reference time"
    name, _plan, visit = found
    label = visit.observation.id.id
    # The execution wait keeps its production length: the COMPLETED is what should end it.
    ctx.odb(
        odb_payload(label, site=name, state="ONGOING", active=ctx.active),
        "UPDATED ONGOING, the expected observation",
    )
    other = await ctx.idle_visit(exclude=(label,), site=name)
    other_label = other[2].observation.id.id if other else label
    if other is None:
        ctx.note("only one visit at the site: the same observation completes")
    ctx.odb(
        odb_payload(other_label, site=name, state="COMPLETED", active=ctx.active),
        "UPDATED COMPLETED while the execution wait is pending",
        shorten={"idle": 1},
    )
    return None


async def c_odb_completed_not_in_plan(ctx: Ctx) -> Optional[str]:
    plans = await ctx.plans()
    if not plans:
        return "no plan in effect"
    name, plan = next(iter(plans.items()))
    label = await ctx.unplanned_label(name, plan)
    ctx.odb(
        odb_payload(label, site=name, state="COMPLETED", active=ctx.active),
        "UPDATED COMPLETED, not in plan",
    )
    return None


async def c_odb_ignored_states(ctx: Ctx) -> Optional[str]:
    found = await ctx.idle_visit()
    if found is None:
        return "no planned visit"
    name, _plan, visit = found
    for state in ("INACTIVE", "UNDEFINED", "UNAPPROVED", "DEFINED"):
        ctx.odb(
            odb_payload(
                visit.observation.id.id, site=name, state=state, active=ctx.active
            ),
            f"UPDATED {state}, in plan",
        )
    return None


def _odb_variant(
    kind: str,
    needs_plan: bool = True,
    edit_type: str = "UPDATED",
    state: str = "READY",
    planned: bool = True,
    **fields,
):
    async def run(ctx: Ctx) -> Optional[str]:
        name, label = "GN", synthetic_label()
        if needs_plan or planned:
            found = await ctx.idle_visit()
            if found is None:
                return "no planned visit"
            name, plan, visit = found
            label = (
                visit.observation.id.id
                if planned
                else await ctx.unplanned_label(name, plan)
            )
        ctx.odb(
            odb_payload(
                label,
                site=name,
                state=state,
                edit_type=edit_type,
                active=ctx.active,
                **fields,
            ),
            kind,
        )
        return None

    return run


async def c_on_demand_twilight(ctx: Ctx) -> Optional[str]:
    await ctx.on_demand(
        ctx.B.OnDemandScheduleEvent(
            site=None, description="On demand request", time=None
        ),
        "schedule_v2 (time=None)",
    )
    return None


async def c_on_demand_wall_clock(ctx: Ctx) -> Optional[str]:
    await ctx.on_demand(
        ctx.B.OnDemandScheduleEvent(
            site=None, description="On demand request", time=datetime.now(UTC)
        ),
        "on_demand_schedule (wall clock)",
    )
    return None


async def c_concurrent(ctx: Ctx) -> Optional[str]:
    plans = await ctx.plans()
    if "GS" not in plans:
        return "needs a plan at GS"
    ctx.weather(weather_payload("GN", 0.7, 0.5), "weather GN, starts a plan")
    if not await ctx.wait_for(lambda: ctx.ledger.planning > 0, 60):
        return "the first plan never started"
    ctx.odb(
        odb_payload(synthetic_label(), site="GS", state="READY", active=ctx.active),
        "UPDATED READY new at GS, during the plan",
    )
    ctx.weather(weather_payload("GS", 0.7, 0.5), "weather GS, during the plan")
    return None


def _fault_case(kind: str, while_down: int):
    async def run(ctx: Ctx) -> Optional[str]:
        feed = ctx.mocks.odb
        if not feed.connected:
            return "the ODB subscription is not connected (an earlier fault killed it?)"
        label = synthetic_label()

        def payload():
            return odb_payload(label, site="GN", state="COMPLETED", active=ctx.active)

        ctx.odb(payload(), "before the fault")
        before = feed.connections
        if kind == "malformed":
            bad = payload()
            bad["obscalcUpdate"]["editType"] = "BOGUS"
            ctx.odb(bad, "malformed frame (fails validation)")
        else:
            feed.fault(kind)
        if not await ctx.wait_for(lambda: not feed.connected, 30):
            return "the subscription never dropped"
        for i in range(while_down):
            ctx.odb(payload(), f"sent while disconnected #{i + 1}")
        if kind == "fatal":
            await asyncio.sleep(2.0)
            if feed.connected:
                ctx.note("the subscription came back after a fatal error")
            return None
        if not await ctx.wait_reconnected(feed, before):
            ctx.note("never reconnected within 45s")
            return None
        down, up = feed.downtimes[-1]
        if up is not None:
            ctx.note(f"reconnected after {up - down:.1f}s")
        ctx.odb(payload(), "after reconnecting")
        return None

    return run


async def c_night_morning(ctx: Ctx) -> Optional[str]:
    for name in ("GS", "GN"):
        ctx.night(ctx.night_event(name, "morning"), f"morning 12° twilight {name}")
    return None


async def c_end_of_night(ctx: Ctx) -> Optional[str]:
    ctx.night(ctx.end_of_night_event(), "End of Night")
    if not await ctx.wait_for(lambda: not ctx.engine_alive, ctx.args.case_timeout):
        ctx.note("the engine was still running after End of Night")
    ctx.weather(weather_payload("GN", 0.7, 0.5), "weather after End of Night")
    return None


def catalog() -> List[Case]:
    """In run order. Cases read the plan in effect, so later ones see what earlier ones did."""
    bug_created = "CREATED+READY raises AttributeError on first_science_target (odb_event_handler.py _on_created_edit)"
    bug_delete = "HARD_DELETE compares a str label with ObservationID, never matches (odb_event_handler.py _on_deleted_edit)"
    return [
        Case(
            "odb_before_plan",
            "odb",
            "ODB UPDATED READY at both sites before any plan exists",
            Expect(),
            c_odb_before_plan,
        ),
        Case(
            "night_evening_twilight",
            "night",
            "Evening 12° twilight GS, GN (seeds the plans)",
            Expect(plans=2),
            c_night_evening,
        ),
        Case(
            "night_non_trigger",
            "night",
            "Sunset, midnight, moonrise: the tracker ignores them",
            Expect(filtered=3),
            c_night_non_trigger,
        ),
        Case(
            "weather_valid_gn",
            "weather",
            "Weather update GN",
            Expect(plans=1),
            _weather_case("GN", 0.7, 0.5),
        ),
        Case(
            "weather_valid_gs",
            "weather",
            "Weather update GS",
            Expect(plans=1),
            _weather_case("GS", 0.7, 0.5),
        ),
        Case(
            "weather_repeat_identical",
            "weather",
            "Same GN update twice (no change check)",
            Expect(plans=2),
            _weather_case("GN", 0.2, 0.5, times=2),
        ),
        Case(
            "weather_bad_cloud_cover",
            "weather",
            "cloudCover 0.33 is not a CloudCover",
            Expect(errors=1),
            _weather_case("GN", 0.7, 0.33),
        ),
        Case(
            "weather_unknown_site",
            "weather",
            "site GX is not a Site",
            Expect(errors=1),
            _weather_case("GX", 0.7, 0.5),
        ),
        Case(
            "odb_ongoing_expected",
            "odb",
            "ONGOING for the observation the plan expects now: wait it out",
            Expect(timer="execution"),
            c_odb_ongoing_expected,
        ),
        Case(
            "odb_ongoing_duplicate",
            "odb",
            "ONGOING twice for the expected observation: the wait restarts",
            Expect(timer="execution"),
            c_odb_ongoing_duplicate,
        ),
        Case(
            "odb_ongoing_out_of_order",
            "odb",
            "ONGOING for an observation planned later",
            Expect(plans=1),
            c_odb_ongoing_out_of_order,
        ),
        Case(
            "odb_ongoing_not_in_plan",
            "odb",
            "ONGOING for an observation not in the plan",
            Expect(plans=1),
            c_odb_ongoing_not_in_plan,
        ),
        Case(
            "odb_ready_in_plan_tight",
            "odb",
            "READY edit of a planned observation, tightest constraints",
            Expect(plans=1),
            _ready_in_plan(TIGHT),
        ),
        Case(
            "odb_ready_in_plan_loose",
            "odb",
            "READY edit of a planned observation, loosest constraints",
            Expect(plans=1),
            _ready_in_plan(LOOSE),
        ),
        Case(
            "odb_ready_burst",
            "odb",
            "Three quick READY edits of one planned observation",
            Expect(plans=3),
            _ready_in_plan(LOOSE, times=3),
        ),
        Case(
            "odb_ready_new_observation",
            "odb",
            "READY for an observation not in the plan",
            Expect(plans=1),
            c_odb_ready_new,
        ),
        Case(
            "odb_ready_wrong_site",
            "odb",
            "READY for a planned observation, instrument of the other site",
            Expect(plans=1),
            c_odb_ready_wrong_site,
        ),
        Case(
            "odb_completed_in_plan",
            "odb",
            "COMPLETED for a planned observation: idle watch",
            Expect(timer="idle"),
            c_odb_completed_in_plan,
        ),
        Case(
            "odb_completed_during_execution_wait",
            "odb",
            "ONGOING (execution wait), then COMPLETED at the same site: the idle watch replaces it",
            Expect(timer="idle"),
            c_odb_completed_during_wait,
        ),
        Case(
            "odb_completed_not_in_plan",
            "odb",
            "COMPLETED for an observation not in the plan",
            Expect(),
            c_odb_completed_not_in_plan,
        ),
        Case(
            "odb_ignored_states",
            "odb",
            "INACTIVE, UNDEFINED, UNAPPROVED, DEFINED for a planned observation",
            Expect(),
            c_odb_ignored_states,
            synthetic=True,
        ),
        Case(
            "odb_no_instrument",
            "odb",
            "UPDATED with no instrument",
            Expect(),
            _odb_variant("UPDATED READY, no instrument", instrument=None),
        ),
        Case(
            "odb_reference_missing",
            "odb",
            "UPDATED with reference=None",
            Expect(errors=1),
            _odb_variant("UPDATED READY, reference=None", reference=False),
        ),
        Case(
            "odb_workflow_missing",
            "odb",
            "UPDATED with workflow=None",
            Expect(errors=1),
            _odb_variant("UPDATED, workflow=None", workflow=False),
        ),
        Case(
            "odb_value_missing",
            "odb",
            "UPDATED with value=None",
            Expect(errors=1),
            _odb_variant("UPDATED, value=None", value=False),
        ),
        Case(
            "odb_created_ready",
            "odb",
            "CREATED, READY",
            Expect(plans=1),
            _odb_variant(
                "CREATED READY", needs_plan=False, planned=False, edit_type="CREATED"
            ),
            known_bug=bug_created,
        ),
        Case(
            "odb_created_defined",
            "odb",
            "CREATED, DEFINED",
            Expect(),
            _odb_variant(
                "CREATED DEFINED",
                needs_plan=False,
                planned=False,
                edit_type="CREATED",
                state="DEFINED",
            ),
        ),
        Case(
            "odb_delete_planned",
            "odb",
            "HARD_DELETE of a planned observation",
            Expect(plans=1),
            _odb_variant("HARD_DELETE, in plan", edit_type="HARD_DELETE"),
            known_bug=bug_delete,
        ),
        Case(
            "odb_delete_unplanned",
            "odb",
            "HARD_DELETE of an observation not in the plan",
            Expect(),
            _odb_variant(
                "HARD_DELETE, not in plan", planned=False, edit_type="HARD_DELETE"
            ),
        ),
        Case(
            "odb_delete_value_missing",
            "odb",
            "HARD_DELETE with value=None",
            Expect(errors=1),
            _odb_variant(
                "HARD_DELETE, value=None", edit_type="HARD_DELETE", value=False
            ),
        ),
        Case(
            "resource_disabled",
            "resource",
            "Resource disabled (no production subscription)",
            Expect(),
            lambda ctx: _send_resource(ctx, "disabled"),
            needs_resource=True,
        ),
        Case(
            "resource_enabled",
            "resource",
            "Resource enabled (no production subscription)",
            Expect(),
            lambda ctx: _send_resource(ctx, "enabled"),
            needs_resource=True,
        ),
        Case(
            "resource_unknown_status",
            "resource",
            "Resource status 'broken'",
            Expect(errors=1),
            lambda ctx: _send_resource(ctx, "broken"),
            needs_resource=True,
        ),
        Case(
            "on_demand_from_twilight",
            "on_demand",
            "schedule_v2: OnDemandScheduleEvent(time=None)",
            Expect(plans=1),
            c_on_demand_twilight,
        ),
        Case(
            "on_demand_wall_clock",
            "on_demand",
            "on_demand_schedule: wall-clock time against a simulated night (recorded only)",
            Expect(plans=1, record_only=True),
            c_on_demand_wall_clock,
        ),
        Case(
            "concurrent_during_plan",
            "concurrency",
            "Weather GN, then ODB GS + weather GS while it plans",
            Expect(plans=3),
            c_concurrent,
        ),
        Case(
            "fault_connection_closed",
            "fault",
            "ODB socket drops; two events sent while it is down",
            Expect(lost=2),
            _fault_case("closed", 2),
        ),
        Case(
            "fault_stream_end",
            "fault",
            "ODB server completes the stream; one event sent before it is back",
            Expect(lost=1),
            _fault_case("end", 1),
        ),
        Case(
            "fault_malformed_odb",
            "fault",
            "A frame that fails validation tears the subscription down",
            Expect(lost=1),
            _fault_case("malformed", 0),
        ),
        Case(
            "night_morning_twilight",
            "night",
            "Morning 12° twilight GS, GN",
            Expect(plans=2),
            c_night_morning,
        ),
        Case(
            "fault_fatal_odb",
            "fault",
            "Non-retryable error: the ODB subscription is gone for the night",
            Expect(lost=1),
            _fault_case("fatal", 1),
        ),
        Case(
            "end_of_night",
            "night",
            "End of Night stops the engine; a later event is never planned",
            Expect(not_published=1, stranded=1),
            c_end_of_night,
        ),
    ]


async def _send_resource(ctx: Ctx, status: str) -> Optional[str]:
    ctx.resource(resource_payload(status), f"resource {status}")
    return None


SEED_CASE = "night_evening_twilight"


def _select_cases(text: Optional[str]) -> List[Case]:
    cases = catalog()
    if not text:
        return cases
    wanted = {t.strip() for t in text.split(",") if t.strip()}
    known = {c.name for c in cases} | {c.group for c in cases}
    unknown = sorted(wanted - known)
    if unknown:
        sys.exit(f"Unknown case or group: {unknown}. `cases --list` shows them.")
    chosen = [c for c in cases if c.name in wanted or c.group in wanted]
    # Everything but odb_before_plan needs plans in effect: seed them first.
    if any(c.name != "odb_before_plan" for c in chosen) and all(
        c.name != SEED_CASE for c in chosen
    ):
        seed = next(c for c in cases if c.name == SEED_CASE)
        index = 1 if chosen and chosen[0].name == "odb_before_plan" else 0
        chosen.insert(index, seed)
    return chosen


# --------------------------------------------------------------------------------------
# Evaluating a case
# --------------------------------------------------------------------------------------


def _delivered_ok(record: SchedRecord) -> bool:
    return record.t_delivered is not None and not record.delivered_error


def _outcome(ledger: Ledger, case: str) -> Dict[str, Any]:
    injections = ledger.case_injections(case)
    ids = {i.inj_id for i in injections}
    scheds = [s for s in ledger.scheds.values() if s.cause in ids]
    direct = [s for s in scheds if s.origin in DIRECT_ORIGINS]
    timers = [s for s in scheds if s.origin in TIMER_ORIGINS]
    return {
        "plans": sum(_delivered_ok(s) for s in direct),
        "timer": sorted(s.origin[: -len("_timer")] for s in timers if _delivered_ok(s)),
        "errors": sum(1 for i in injections if i.error),
        "lost": sum(1 for i in injections if i.lost),
        "filtered": sum(1 for i in injections if i.filtered),
        "not_published": sum(
            1 for s in direct if s.end_of_night and s.plan_error is None
        ),
        "stranded": sum(1 for s in scheds if s.t_plan_start is None),
        "plan_errors": sum(1 for s in scheds if s.plan_error),
        "unhandled": sum(
            1
            for i in injections
            if i.t_handle_start is None and not i.lost and not i.filtered
        ),
    }


def _expected(expect: Expect) -> Dict[str, Any]:
    return {
        "plans": expect.plans,
        "timer": [expect.timer] if expect.timer else [],
        "errors": expect.errors,
        "lost": expect.lost,
        "filtered": expect.filtered,
        "not_published": expect.not_published,
        "stranded": expect.stranded,
        "plan_errors": 0,
        "unhandled": 0,
    }


def _describe(outcome: Dict[str, Any]) -> str:
    parts = []
    for key, label in (
        ("plans", "plan"),
        ("timer", "timer"),
        ("errors", "error"),
        ("lost", "lost"),
        ("filtered", "filtered"),
        ("not_published", "not published"),
        ("stranded", "stranded"),
        ("plan_errors", "plan error"),
        ("unhandled", "unhandled"),
    ):
        value = outcome[key]
        if key == "timer":
            if value:
                parts.append("timer " + "+".join(value))
        elif value:
            parts.append(f"{label} {value}")
    return ", ".join(parts) or "nothing"


@dataclass
class CaseResult:
    name: str
    group: str
    summary: str
    status: str
    expected: str
    actual: str
    processing_s: Optional[float]
    response_s: Optional[float]
    notes: List[str]


def _case_timings(ledger: Ledger, case: str) -> Tuple[Optional[float], Optional[float]]:
    """Worst processing (received -> schedule event) and response (received -> plan delivered).

    A timer replan's response is counted from when its timer fired: the wait is by design.
    """
    processing, response = [], []
    for inj in ledger.case_injections(case):
        if inj.t_received is None:
            continue
        for sid in inj.sched_ids:
            record = ledger.scheds[sid]
            if record.origin not in DIRECT_ORIGINS:
                continue
            processing.append(record.t_sched - inj.t_received)
            if _delivered_ok(record):
                response.append(record.t_delivered - inj.t_received)
    for record in ledger.scheds.values():
        inj = ledger.injections.get(record.cause or "")
        if (
            inj is not None
            and inj.case == case
            and record.origin in TIMER_ORIGINS
            and _delivered_ok(record)
        ):
            response.append(record.t_delivered - record.t_sched)
    return (
        max(processing) if processing else None,
        max(response) if response else None,
    )


def _evaluate(
    ledger: Ledger,
    case: Case,
    skip: Optional[str],
    settled: bool,
    busy: List[str],
    notes: List[str],
) -> CaseResult:
    expected = _expected(case.expect)
    if skip is not None:
        return CaseResult(
            case.name,
            case.group,
            case.summary,
            "SKIPPED",
            _describe(expected),
            "-",
            None,
            None,
            [skip] + notes,
        )
    actual = _outcome(ledger, case.name)
    diffs = [k for k in expected if expected[k] != actual[k]]
    notes = list(notes)
    if case.synthetic:
        notes.append("synthetic: the real subscription would not send this")
    if case.expect.record_only:
        status = "RECORDED"
    elif not diffs:
        status = "PASS"
        if case.known_bug:
            notes.append("known bug not reproduced (fixed?)")
    elif case.known_bug:
        status = "KNOWN BUG"
        notes.append(case.known_bug)
    else:
        status = "FAIL"
    if not settled:
        notes.append(
            f"did not settle within the case timeout (busy: {', '.join(busy)})"
        )
        if status == "PASS":
            status = "FAIL"
    case_ids = {i.inj_id for i in ledger.case_injections(case.name)}
    errors = sorted({i.error for i in ledger.case_injections(case.name) if i.error})
    plan_errors = sorted(
        {
            s.plan_error
            for s in ledger.scheds.values()
            if s.plan_error and s.cause in case_ids
        }
    )
    notes += [f"handler: {e}" for e in errors[:2]] + [
        f"plan: {e}" for e in plan_errors[:2]
    ]
    processing, response = _case_timings(ledger, case.name)
    return CaseResult(
        case.name,
        case.group,
        case.summary,
        status,
        _describe(expected),
        _describe(actual),
        processing,
        response,
        notes,
    )


# --------------------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------------------


async def _run_cases(ctx: Ctx, cases: List[Case]) -> List[CaseResult]:
    results = []
    for case in cases:
        ctx.case, ctx.notes = case.name, []
        print(f"  {case.name}: {case.summary}...", flush=True)
        try:
            skip = await case.run(ctx)
        except Exception as exc:
            skip = f"harness error: {type(exc).__name__}: {rt._short(exc)}"
        settled, busy = (
            (True, []) if skip else await ctx.wait_settled(ctx.args.case_timeout)
        )
        result = _evaluate(ctx.ledger, case, skip, settled, busy, ctx.notes)
        results.append(result)
        print(
            f"    {result.status}: expected {result.expected}; got {result.actual}"
            + (f" ({'; '.join(result.notes)})" if result.notes else ""),
            flush=True,
        )
    return results


LOAD_KINDS = {
    "weather": "plan",
    "odb_ready_in_plan": "plan",
    "odb_ongoing_out_of_order": "plan",
    "odb_ready_new": "plan",
    "odb_completed_not_in_plan": "none",
}


def _parse_mix(text: str) -> Dict[str, float]:
    mix = {}
    for part in text.split(","):
        if not part.strip():
            continue
        name, _, weight = part.partition("=")
        name = name.strip()
        if name not in LOAD_KINDS:
            sys.exit(f"--mix: unknown kind {name!r}; known: {', '.join(LOAD_KINDS)}")
        try:
            mix[name] = float(weight)
        except ValueError:
            sys.exit(f"--mix: weight for {name} must be a number, got {weight!r}")
    if not mix or sum(mix.values()) <= 0:
        sys.exit("--mix needs at least one positive weight.")
    return mix


class _PlanSnapshot:
    """The plan in effect, re-read every few seconds instead of on every load event: each read
    deep-copies the plan under the store lock, and that cost belongs to the handlers."""

    def __init__(self, ctx: Ctx) -> None:
        self.ctx = ctx
        self.planned: Dict[
            str, List[Tuple[str, bool, bool]]
        ] = {}  # site -> (label, ready, now)
        self.unplanned: Dict[str, str] = {}

    async def refresh(self) -> None:
        planned, unplanned = {}, {}
        for name, plan in (await self.ctx.plans()).items():
            when = await self.ctx.when(name)
            planned[name] = [
                (
                    v.observation.id.id,
                    v.observation.status is self.ctx.B.ObservationStatus.READY,
                    v.start_time <= when < self.ctx.visit_end(plan, v),
                )
                for v in plan.visits
            ]
            unplanned[name] = await self.ctx.unplanned_label(name, plan)
        self.planned, self.unplanned = planned, unplanned

    async def run(self) -> None:
        while True:
            try:
                await self.refresh()
            except Exception:
                pass
            await asyncio.sleep(5.0)


def _emit_load_event(
    ctx: Ctx, snapshot: _PlanSnapshot, kind: str, rng: random.Random
) -> Injection:
    if kind == "odb_ready_in_plan":
        options = [
            (s, label)
            for s, rows in snapshot.planned.items()
            for label, ready, _now in rows
            if ready
        ]
        if not options:
            kind = "odb_ready_new"
    if kind == "odb_ongoing_out_of_order":
        options = [
            (s, label)
            for s, rows in snapshot.planned.items()
            for label, _ready, now in rows
            if not now
        ]
        if not options:
            kind = "odb_ready_new"
    if kind == "odb_completed_not_in_plan" or kind == "odb_ready_new":
        if not snapshot.unplanned:
            kind = "weather"
    if kind == "weather":
        site = rng.choice(SITE_NAMES)
        inj = ctx.weather(
            weather_payload(site, rng.choice((0.2, 0.7)), rng.choice((0.5, 0.7))), kind
        )
    elif kind in ("odb_ready_in_plan", "odb_ongoing_out_of_order"):
        site, label = rng.choice(options)
        state = "READY" if kind == "odb_ready_in_plan" else "ONGOING"
        inj = ctx.odb(
            odb_payload(label, site=site, state=state, active=ctx.active), kind
        )
    else:
        site = rng.choice(sorted(snapshot.unplanned))
        label = (
            synthetic_label()
            if kind == "odb_ready_new" and ctx.args.visibility == "sight"
            else snapshot.unplanned[site]
        )
        state = "READY" if kind == "odb_ready_new" else "COMPLETED"
        inj = ctx.odb(
            odb_payload(label, site=site, state=state, active=ctx.active), kind
        )
    inj.expect = LOAD_KINDS[kind]
    return inj


async def _run_load(ctx: Ctx, args) -> Dict[str, Any]:
    print("  seeding: evening twilight GS, GN...", flush=True)
    ctx.case = SEED_CASE
    await c_night_evening(ctx)
    settled, busy = await ctx.wait_settled(args.case_timeout)
    if not settled:
        print(
            f"  seed did not settle (busy: {', '.join(busy)}); continuing", flush=True
        )

    snapshot = _PlanSnapshot(ctx)
    await snapshot.refresh()
    refresher = asyncio.create_task(snapshot.run())
    mix = _parse_mix(args.mix)
    kinds, weights = list(mix), list(mix.values())
    rng = random.Random(args.seed)
    ctx.case = "load"
    started = time.monotonic()
    sent = odb_sent = drops = 0
    print(
        f"  load: {'burst of ' + str(args.burst) if args.burst else f'{args.rate}/s'}"
        f"{'' if args.burst else f' for up to {args.duration}s'}"
        f"{f' or {args.count} events' if args.count else ''}...",
        flush=True,
    )
    try:
        while True:
            if args.burst and sent >= args.burst:
                break
            if not args.burst and (
                time.monotonic() - started >= args.duration
                or (args.count and sent >= args.count)
            ):
                break
            kind = rng.choices(kinds, weights)[0]
            inj = _emit_load_event(ctx, snapshot, kind, rng)
            sent += 1
            if inj.source == "odb":
                odb_sent += 1
                if (
                    args.drop_every
                    and odb_sent % args.drop_every == 0
                    and ctx.mocks.odb.fault("closed")
                ):
                    drops += 1
            if not args.burst:
                await asyncio.sleep(rng.expovariate(args.rate))
    finally:
        refresher.cancel()
    injected_for = time.monotonic() - started
    sending_ended = ctx.ledger.now()
    print(
        f"  sent {sent} events in {injected_for:.0f}s; draining (up to {args.drain_timeout}s)...",
        flush=True,
    )
    settled, busy = await ctx.wait_settled(args.drain_timeout)
    return {
        "sent": sent,
        "drops": drops,
        "injected_for_s": injected_for,
        "settled": settled,
        "busy": busy,
        "drain_s": ctx.ledger.now() - sending_ended,
        "mix": mix,
    }


# --------------------------------------------------------------------------------------
# Setup and teardown of one measured run
# --------------------------------------------------------------------------------------


def _night_events(B: SimpleNamespace, night_start: datetime) -> Dict[str, list]:
    """The simulated night's phenomena per site. The 08:00 UT anchor lands in the right local
    night for both sites (before noon in Chile, after noon in Hawaii)."""
    return {
        name: B.NightTracker.calculate_night_events(night_start, B.Site[name])
        for name in SITE_NAMES
    }


def _simulated_now(night_events: Dict[str, list], value: str) -> datetime:
    evening = [events[NIGHT_INDEX["evening"]].time for events in night_events.values()]
    morning = [events[NIGHT_INDEX["morning"]].time for events in night_events.values()]
    if value == "overlap":
        # Both sites dark, so the timer cases can find a running visit at either.
        start = max(evening) + timedelta(minutes=30)
        if start <= min(morning) - timedelta(hours=1):
            return start
        return min(evening) + timedelta(hours=1)
    try:
        return min(evening) + timedelta(hours=float(value))
    except ValueError:
        sys.exit(
            f"--sim-now must be 'overlap' or hours after the first evening twilight, got {value!r}"
        )


async def _measure(args) -> int:
    B = _backend()
    from lucupy.observatory.abstract import ObservatoryProperties
    from lucupy.observatory.gemini.geminiproperties import GeminiProperties
    from scheduler.clients.gpp import gpp
    from scheduler.services.telemetry import otel
    from scheduler.services.telemetry.perf import PERF_LOGGER_NAME

    ObservatoryProperties.set_properties(GeminiProperties)
    night_start = rt._night_anchor(args.night)
    vis_end = rt._vis_end(night_start, args.vis_end)
    rt._check_ops_calendar(night_start, vis_end)
    night_events = await asyncio.to_thread(_night_events, B, night_start)
    sim_now = _simulated_now(night_events, args.sim_now)
    night_date = night_start.date()
    active = (
        (night_date - timedelta(days=1)).isoformat(),
        (night_date + timedelta(days=1)).isoformat(),
    )

    cases = _select_cases(args.cases) if args.mode == "cases" else []
    resource_subscription = any(c.needs_resource for c in cases)
    print(
        f"Night {night_date} (anchor {night_start:%Y-%m-%d %H:%M} UT); simulated now {sim_now:%Y-%m-%d %H:%M} UT; "
        f"engine reads {len(rt._window_nights(night_start, vis_end))} nights."
    )

    sight = args.visibility == "sight"
    if sight:
        from scheduler.services.sight.database.connection import (
            dispose_engine,
            init_db_engine,
        )

        await init_db_engine()

    meta: Dict[str, Any] = {}
    payload: Optional[List[dict]] = None
    program_ids: Optional[List[str]] = None
    if args.source == "clones":
        sources = await rt._fetch_sources(args.programs)
        meta = sources["meta"]
        clones = rt.build_clone_set(
            sources,
            night_start,
            [args.size],
            args.cap,
            args.seed,
            args.allow_nonsidereal,
        )
        payload = [clones.payloads[j] for j in clones.selections[args.size]]
        meta["clone_totals"] = clones.totals[args.size]
        print(
            f"Clones: {len(payload)} programs, {clones.totals[args.size]} observations per site."
        )
    elif args.programs:
        program_ids = await rt._resolve_program_ids(args.programs)

    ledger = Ledger()
    ledger.timer_wait = args.timer_wait
    mocks = MockSources(B, ledger, args.iq, args.cc, resource_subscription)
    patches = rt._Patches()
    recorder = rt._Recorder()
    perf_logger = logging.getLogger(PERF_LOGGER_NAME)
    saved_handlers = list(perf_logger.handlers)
    capture = rt._PerfCapture(recorder)
    reader, exporting = rt._setup_metrics()
    started_at = datetime.now(UTC)
    sub_queue: asyncio.Queue = asyncio.Queue()
    proc = None
    harness_tasks: List[asyncio.Task] = []
    results: List[CaseResult] = []
    load_info: Optional[Dict[str, Any]] = None
    worst_stall = None

    async def fake_gpp_program_data(program_list=None):
        # Parsing mutates the payload, so every plan gets its own copy, made off the loop so it
        # does not show up as a loop stall the monitor never has in production.
        t = time.perf_counter()
        items = await asyncio.to_thread(copy.deepcopy, payload)
        ledger.payload_copy_s.append(time.perf_counter() - t)

        async def generate():
            for item in items:
                yield item

        return generate()

    try:
        if payload is not None:
            patches.set(B.simulationbuilder, "gpp_program_data", fake_gpp_program_data)
        _install_hooks(patches, B, ledger, mocks, args.timer_mode)
        rt._quiet_scheduler_logs(args.log_level)
        for handler in saved_handlers:
            perf_logger.removeHandler(handler)
        perf_logger.addHandler(capture)

        await B.build_params_store.set(
            B.BuildParameters(
                visibility_start=night_start,
                visibility_end=vis_end,
                program_list=program_ids,
                simulated_now=sim_now,
            )
        )
        params = B.SchedulerParameters(
            start=night_start,
            end=night_start + timedelta(days=rt.NIGHTS_IN_WINDOW - 1),
            sites=B.ALL_SITES,
            mode=B.SchedulerModes.OPERATION,
            semester_visibility=False,
            num_nights_to_schedule=1,
        )
        proc = B.SchedulerProcess(PROCESS_ID, params)
        B.plan_response_subscribers.setdefault(PROCESS_ID, set()).add(sub_queue)
        harness_tasks.append(
            asyncio.create_task(_collect_deliveries(sub_queue, B, ledger))
        )
        await proc.run()
        harness_tasks.append(
            asyncio.create_task(
                _sample(ledger, proc.night_monitor.event_queue, proc.scheduler_queue)
            )
        )

        ctx = Ctx(args, B, ledger, mocks, proc, night_events, active)
        if not await ctx.wait_connected(mocks.feeds()):
            raise RuntimeError("the mock subscriptions never connected")
        if not await ctx.wait_for(lambda: mocks.night_queue is not None, 10):
            raise RuntimeError("the mock night clock never started")

        if args.mode == "cases":
            print(f"Running {len(cases)} case(s)...", flush=True)
            results = await _run_cases(ctx, cases)
        else:
            load_info = await _run_load(ctx, args)

        # Never stop the engine mid-plan: run_rt's thread would keep the process alive.
        await ctx.wait_for(lambda: ledger.planning == 0, args.case_timeout)
        worst_stall = proc.night_monitor.loop_monitor.worst_stall
    finally:
        if proc is not None and proc.night_monitor is not None:
            consumer = proc.night_monitor.consumer
            # NightMonitor.shutdown touches neither, so their tasks would outlive it.
            try:
                await consumer.odb_handler.on_timeline_reset()
                await consumer.resource_handler.pdr.clear()
            except Exception:
                pass
        if proc is not None:
            await proc.stop_process()
        for task in harness_tasks:
            task.cancel()
        await asyncio.gather(*harness_tasks, return_exceptions=True)
        B.plan_response_subscribers.get(PROCESS_ID, set()).discard(sub_queue)
        patches.undo()
        perf_logger.removeHandler(capture)
        for handler in saved_handlers:
            perf_logger.addHandler(handler)
        _record_otel(ledger, otel)
        otel.shutdown_telemetry()
        await gpp.close()
        if sight:
            await dispose_engine()

    meta.update(
        {
            "label": args.label,
            "mode": args.mode,
            "started_at": started_at.isoformat(timespec="seconds"),
            "night": night_date.isoformat(),
            "simulated_now": sim_now.isoformat(timespec="minutes"),
            "vis_end": vis_end.date().isoformat(),
            "nights": len(rt._window_nights(night_start, vis_end)),
            "source": args.source,
            "size": args.size if args.source == "clones" else None,
            "visibility": args.visibility,
            "timer_mode": args.timer_mode,
            "timer_wait_s": args.timer_wait,
            "worst_loop_stall_s": worst_stall,
            "weather_reads": mocks.weather_reads,
            "reconnects": {f.name: f.connections - 1 for f in mocks.feeds()},
            "exporting": exporting,
        }
    )
    status = _write_outputs(args, meta, ledger, results, load_info, recorder)
    return 0 if status == "OK" else 1


# --------------------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------------------

STAGES = (
    ("receive", "mock send -> listener receives (loop lag)"),
    ("queue_wait", "receive -> handler starts (listener put + event queue)"),
    ("handler", "handler runs"),
    ("processing", "receive -> schedule event sent"),
    ("sched_wait", "schedule event -> engine starts it"),
    ("plan", "engine plan"),
    ("publish", "plan done -> subscriber has it"),
    ("response", "receive -> plan delivered"),
)


def _stage_values(ledger: Ledger) -> Dict[str, Dict[str, List[float]]]:
    """stage -> source -> durations. Plan stages count once per schedule event, against the
    source of the event that caused it."""
    out: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    def add(stage: str, source: str, value: Optional[float]) -> None:
        if value is not None:
            out[stage][source].append(value)
            out[stage]["all"].append(value)

    for inj in ledger.injections.values():
        if inj.t_received is None:
            continue
        if inj.source in ("odb", "weather", "resource"):
            add("receive", inj.source, inj.t_received - inj.t_sent)
        if inj.t_handle_start is not None and inj.source in (
            "odb",
            "weather",
            "resource",
        ):
            add("queue_wait", inj.source, inj.t_handle_start - inj.t_received)
        if inj.t_handle_start is not None and inj.t_handle_end is not None:
            add("handler", inj.source, inj.t_handle_end - inj.t_handle_start)
        direct = [
            ledger.scheds[s]
            for s in inj.sched_ids
            if ledger.scheds[s].origin in DIRECT_ORIGINS
        ]
        if direct:
            add("processing", inj.source, direct[0].t_sched - inj.t_received)
        for record in direct:
            if _delivered_ok(record):
                add("response", inj.source, record.t_delivered - inj.t_received)
    for record in ledger.scheds.values():
        inj = ledger.injections.get(record.cause or "")
        source = (
            inj.source
            if inj is not None and record.origin in DIRECT_ORIGINS
            else record.origin
        )
        if record.t_plan_start is not None:
            add("sched_wait", source, record.t_plan_start - record.t_sched)
        if record.t_plan_start is not None and record.t_plan_end is not None:
            add("plan", source, record.t_plan_end - record.t_plan_start)
        if record.t_plan_end is not None and record.t_delivered is not None:
            add("publish", source, record.t_delivered - record.t_plan_end)
    return out


def _pct(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[
        min(len(ordered) - 1, max(0, int(round(p / 100 * len(ordered) + 0.5)) - 1))
    ]


def _record_otel(ledger: Ledger, otel) -> None:
    """Stage durations into the same histogram `timed` uses, as stress.nm.<stage>."""
    try:
        histogram = otel.telemetry.operation_duration
        for stage, by_source in _stage_values(ledger).items():
            for value in by_source.get("all", []):
                histogram.record(value, {"operation": f"stress.nm.{stage}", "ok": True})
    except Exception:
        pass


def _accounting(ledger: Ledger) -> Dict[str, Any]:
    inj = list(ledger.injections.values())
    scheds = list(ledger.scheds.values())
    received = [i for i in inj if i.t_received is not None]
    lost = Counter(i.lost for i in inj if i.lost)
    handled_ok = [
        i
        for i in received
        if i.t_handle_end is not None and not i.error and not i.filtered
    ]
    errors = [i for i in received if i.error]
    filtered = [i for i in received if i.filtered]
    unhandled = [i for i in received if i.t_handle_start is None]
    in_handler = [
        i for i in received if i.t_handle_start is not None and i.t_handle_end is None
    ]
    never = [i for i in inj if i.t_received is None and not i.lost]
    planned = [s for s in scheds if s.t_plan_start is not None]
    stranded = [s for s in scheds if s.t_plan_start is None]
    plan_ok = [s for s in planned if s.plan_error is None and s.t_plan_end is not None]
    plan_err = [s for s in planned if s.plan_error is not None]
    not_published = [s for s in plan_ok if s.end_of_night]
    delivered = [
        s for s in scheds if s.t_delivered is not None and not s.delivered_error
    ]
    delivered_err = [s for s in scheds if s.delivered_error]
    undelivered = [s for s in plan_ok if not s.end_of_night and s.t_delivered is None]
    with_expect = [i for i in inj if i.expect]
    # Only events the monitor actually received: what a dropped socket ate is counted as lost.
    missing = [
        i
        for i in with_expect
        if i.expect == "plan"
        and i.t_received is not None
        and not any(_delivered_ok(ledger.scheds[s]) for s in i.sched_ids)
    ]
    unexpected = [i for i in with_expect if i.expect == "none" and i.sched_ids]
    return {
        "injected": len(inj),
        "received": len(received),
        "lost": dict(lost),
        "lost_total": sum(lost.values()),
        "never_received": len(never),
        "handled_ok": len(handled_ok),
        "handler_errors": len(errors),
        "filtered": len(filtered),
        "unhandled": len(unhandled),
        "in_handler": len(in_handler),
        "schedule_events": len(scheds),
        "by_origin": dict(Counter(s.origin for s in scheds)),
        "planned": len(planned),
        "stranded": len(stranded),
        "plan_ok": len(plan_ok),
        "plan_errors": len(plan_err),
        "not_published": len(not_published),
        "delivered": len(delivered),
        "delivered_errors": len(delivered_err),
        "undelivered": len(undelivered),
        "unmatched_deliveries": len(ledger.unmatched_deliveries),
        "expected_plan_missing": len(missing),
        "unexpected_plan": len(unexpected),
        "load_expect_counts": dict(Counter(i.expect for i in with_expect)),
    }


def _conservation(acc: Dict[str, Any]) -> List[Tuple[str, bool]]:
    return [
        (
            "injected = received + lost + never received",
            acc["injected"]
            == acc["received"] + acc["lost_total"] + acc["never_received"],
        ),
        (
            "received = handled + handler error + filtered + unhandled + still in handler",
            acc["received"]
            == acc["handled_ok"]
            + acc["handler_errors"]
            + acc["filtered"]
            + acc["unhandled"]
            + acc["in_handler"],
        ),
        (
            "schedule events = planned + stranded",
            acc["schedule_events"] == acc["planned"] + acc["stranded"],
        ),
        (
            "planned = delivered + plan error + not published + undelivered",
            acc["planned"]
            == acc["delivered"]
            + acc["plan_errors"]
            + acc["not_published"]
            + acc["undelivered"],
        ),
        (
            "plan errors were all published as errors",
            acc["plan_errors"] == acc["delivered_errors"],
        ),
    ]


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


def _row_stats(values: List[float]) -> List[str]:
    return [
        str(len(values)),
        rt._fmt_s(_pct(values, 50)),
        rt._fmt_s(_pct(values, 95)),
        rt._fmt_s(max(values) if values else None),
    ]


def _render(
    args,
    meta: Dict[str, Any],
    ledger: Ledger,
    results: List[CaseResult],
    load_info: Optional[Dict[str, Any]],
    recorder,
) -> Tuple[str, Dict[str, Any]]:
    acc = _accounting(ledger)
    checks = _conservation(acc)
    stages = _stage_values(ledger)
    notes = []
    failed = [r for r in results if r.status == "FAIL"]
    if failed:
        notes.append(
            f"{len(failed)} case(s) failed: {', '.join(r.name for r in failed)}"
        )
    if not all(ok for _name, ok in checks):
        notes.append("accounting does not add up")
    if load_info is not None:
        if not load_info["settled"]:
            notes.append(
                f"did not drain within {args.drain_timeout}s (busy: {', '.join(load_info['busy'])})"
            )
        if acc["expected_plan_missing"]:
            notes.append(
                f"{acc['expected_plan_missing']} event(s) that should have produced a plan did not"
            )
    if ledger.unmatched_deliveries:
        notes.append(
            f"{len(ledger.unmatched_deliveries)} delivery(ies) matched no plan (engine crash?)"
        )
    status = "FAILED" if notes else "OK"
    known = [r for r in results if r.status == "KNOWN BUG"]

    out: List[str] = [f"## {args.label}\n"]
    out.append(f"**Status: {status}**" + (": " + "; ".join(notes) if notes else ""))
    if known:
        out.append(f"\nKnown bugs reproduced: {', '.join(r.name for r in known)}.")
    out.append("")
    out.append(f"- Mode: **{meta['mode']}**; run {meta['started_at']}")
    out.append(
        f"- Night {meta['night']}, simulated now {meta['simulated_now']} UT; engine reads "
        f"{meta['nights']} nights (to {meta['vis_end']}); visibility **{meta['visibility']}**"
    )
    if meta["source"] == "clones":
        out.append(
            f"- Data: clones, {meta.get('clone_totals')} observations per site, from "
            f"{meta.get('programs')} programs fetched at {meta.get('fetched_at')}"
        )
    else:
        out.append("- Data: live ODB, fetched inside every plan")
    out.append(
        f"- Timers: {meta['timer_mode']} (idle {'6 h' if meta['timer_mode'] == 'quiet' else '10 min'}); "
        f"timer cases shorten their own timer to {meta['timer_wait_s']:g}s"
    )
    out.append(
        f"- Subscription reconnects: {meta['reconnects']}; engine weather reads: {meta['weather_reads']}"
    )
    if load_info is not None:
        mix = ", ".join(f"{k} {v:g}" for k, v in load_info["mix"].items())
        out.append(
            f"- Load: {load_info['sent']} events in {load_info['injected_for_s']:.0f}s "
            f"({'burst' if args.burst else f'{args.rate}/s Poisson'}; mix {mix}); "
            f"{load_info['drops']} forced ODB drops; drained in {rt._fmt_s(load_info['drain_s'])}"
        )
    if meta.get("exporting"):
        out.append("- Metrics also pushed over OTLP as deployment.environment=stress")
    out.append("")

    if results:
        out.append("### Cases\n")
        rows = [
            [
                f"`{r.name}`",
                r.expected,
                r.actual,
                f"**{r.status}**" if r.status in ("FAIL", "KNOWN BUG") else r.status,
                rt._fmt_s(r.processing_s),
                rt._fmt_s(r.response_s),
                "; ".join(r.notes)[:300],
            ]
            for r in results
        ]
        out.append(
            rt._table(
                [
                    "case",
                    "expected",
                    "actual",
                    "result",
                    "processing",
                    "response",
                    "notes",
                ],
                rows,
            )
        )
        out.append("")
        out.append(
            "Processing: the event reaches the listener -> schedule event sent (worst event in the "
            "case). Response: the event reaches the listener -> plan delivered; for a timer replan, "
            "the timer firing -> plan delivered."
        )
        out.append("")

    out.append("### Latency per stage\n")
    rows = []
    for stage, description in STAGES:
        by_source = stages.get(stage, {})
        for source in ["all"] + sorted(k for k in by_source if k != "all"):
            values = by_source.get(source, [])
            if values:
                rows.append(
                    [
                        f"`{stage}`" if source == "all" else "",
                        description if source == "all" else "",
                        source,
                    ]
                    + _row_stats(values)
                )
    out.append(rt._table(["stage", "what", "source", "n", "p50", "p95", "max"], rows))
    out.append("")

    out.append("### Accounting\n")
    lost = ", ".join(f"{k} {v}" for k, v in sorted(acc["lost"].items())) or "0"
    rows = [
        ["events injected", str(acc["injected"])],
        ["received by the listener", str(acc["received"])],
        ["lost before a handler (subscription down / bad frame)", lost],
        ["handled, no error", str(acc["handled_ok"])],
        ["handler errors (consumer skipped them)", str(acc["handler_errors"])],
        ["night events filtered by the tracker", str(acc["filtered"])],
        ["received but never handled", str(acc["unhandled"] + acc["in_handler"])],
        [
            "schedule events sent",
            f"{acc['schedule_events']} ({', '.join(f'{k} {v}' for k, v in sorted(acc['by_origin'].items()))})",
        ],
        ["schedule events never planned (stranded)", str(acc["stranded"])],
        [
            "plans computed / failed / end of night (not published)",
            f"{acc['plan_ok']} / {acc['plan_errors']} / {acc['not_published']}",
        ],
        [
            "plans delivered / errors delivered / never delivered",
            f"{acc['delivered']} / {acc['delivered_errors']} / {acc['undelivered']}",
        ],
        ["deliveries matching no plan", str(acc["unmatched_deliveries"])],
    ]
    if acc["load_expect_counts"]:
        # Load events are built from a plan snapshot up to 5 s old: an ONGOING picked as "planned
        # later" can be the expected one by the time it is handled, which waits rather than plans.
        rows.append(
            [
                "load events received that should plan but did not",
                str(acc["expected_plan_missing"]),
            ]
        )
        rows.append(
            ["load events that should not plan but did", str(acc["unexpected_plan"])]
        )
    out.append(rt._table(["", "count"], rows))
    out.append("")
    out.append(
        "Checks: "
        + "; ".join(f"{name}: {'OK' if ok else 'MISMATCH'}" for name, ok in checks)
        + "."
    )
    out.append("")

    out.append("### Backlog and loop\n")
    arms = ledger.timer_arms
    rows = [
        ["max event queue (listener -> consumer)", str(ledger.max_event_queue)],
        ["max scheduler queue (handlers -> engine)", str(ledger.max_sched_queue)],
        ["worst event loop stall", rt._fmt_s(meta.get("worst_loop_stall_s"))],
        [
            "timer arms: idle / execution / shortened",
            f"{sum(a['kind'] == 'idle' for a in arms)} / {sum(a['kind'] == 'execution' for a in arms)} / "
            f"{sum(a['shortened'] for a in arms)}",
        ],
        [
            "idle-timer arms from plan_published",
            str(sum(a["origin"] == "plan_published" for a in arms)),
        ],
        [
            "payload copy per plan (harness, off the loop)",
            rt._fmt_s(
                sum(ledger.payload_copy_s) / len(ledger.payload_copy_s)
                if ledger.payload_copy_s
                else None
            ),
        ],
    ]
    out.append(rt._table(["", "value"], rows))
    out.append("")
    if load_info is not None and ledger.samples:
        out.append("Scheduler queue over time (sampled):\n")
        step = max(1, len(ledger.samples) // 15)
        rows = [[f"{t:.1f}s", str(eq), str(sq)] for t, eq, sq in ledger.samples[::step]]
        out.append(rt._table(["t", "event queue", "scheduler queue"], rows))
        out.append("")

    out.append("### Engine\n")
    planned = sorted(
        (s for s in ledger.scheds.values() if s.plan_index is not None),
        key=lambda s: s.plan_index,
    )
    rows = []
    for start in range(0, len(planned), 10):
        chunk = planned[start : start + 10]
        plan_times = [
            s.t_plan_end - s.t_plan_start for s in chunk if s.t_plan_end is not None
        ]
        waits = [s.t_plan_start - s.t_sched for s in chunk]
        rows.append(
            [
                f"{chunk[0].plan_index}-{chunk[-1].plan_index}",
                rt._fmt_s(sum(plan_times) / len(plan_times) if plan_times else None),
                rt._fmt_s(max(plan_times) if plan_times else None),
                rt._fmt_s(sum(waits) / len(waits) if waits else None),
                ", ".join(
                    f"{k} {v}"
                    for k, v in Counter(s.event_type for s in chunk).most_common(3)
                ),
            ]
        )
    if rows:
        out.append(
            rt._table(
                ["plans", "mean plan", "max plan", "mean queue wait", "events"], rows
            )
        )
        out.append("")
        out.append(
            "Plan time is shown against plan index: stats and serialization run over the whole "
            "stitched timeline, so later plans of a night cost more."
        )
        out.append("")
    ops = defaultdict(list)
    for record in planned:
        for op, duration in recorder.events.get(record.sched_id, []):
            ops[op].append(duration)
    rows = [
        [
            f"`{op}`",
            str(len(ops[op])),
            rt._fmt_s(sum(ops[op]) / len(ops[op])),
            rt._fmt_s(max(ops[op])),
        ]
        for op in ("engine.plan", "scp.build_collector", "scp.select", "scp.optimize")
        if ops.get(op)
    ]
    if rows:
        out.append(rt._table(["operation", "n", "mean", "max"], rows))
        out.append("")

    def p95(stage: str) -> Optional[float]:
        return _pct(stages.get(stage, {}).get("all", []), 95)

    plan_times = stages.get("plan", {}).get("all", [])
    data = {
        "label": args.label,
        "mode": meta["mode"],
        "status": status,
        "events": acc["injected"],
        "lost": acc["lost_total"],
        "errors": acc["handler_errors"],
        "stranded": acc["stranded"],
        "p95_processing": p95("processing"),
        "p95_response": p95("response"),
        "max_sched_queue": ledger.max_sched_queue,
        "plans": acc["plan_ok"],
        "plan_mean": sum(plan_times) / len(plan_times) if plan_times else None,
        "cases": dict(Counter(r.status for r in results)),
    }
    out.append(f"<!-- nm-data {json.dumps(data)} -->")
    return "\n".join(out), data


_SECTION = re.compile(
    r"<!-- section:(?P<label>[A-Za-z0-9_.-]+) -->\n(?P<body>.*?)\n<!-- /section:(?P=label) -->",
    re.DOTALL,
)
_DATA = re.compile(r"<!-- nm-data (\{.*?\}) -->")


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
    out = [
        "# Night monitor stress test",
        "",
        f"_Updated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}. Mock events through the real "
        f"night monitor and the real engine (scripts/stress_night_monitor.py)._",
        "",
    ]
    if data:
        out.append("## Comparison\n")
        rows = [
            [
                d["label"],
                d["mode"],
                d["status"],
                str(d["events"]),
                str(d["lost"]),
                str(d["errors"]),
                str(d["stranded"]),
                rt._fmt_s(d.get("p95_processing")),
                rt._fmt_s(d.get("p95_response")),
                str(d.get("max_sched_queue")),
                str(d.get("plans")),
                rt._fmt_s(d.get("plan_mean")),
            ]
            for d in data
        ]
        out.append(
            rt._table(
                [
                    "label",
                    "mode",
                    "status",
                    "events",
                    "lost",
                    "handler errors",
                    "stranded",
                    "p95 processing",
                    "p95 response",
                    "max sched queue",
                    "plans",
                    "mean plan",
                ],
                rows,
            )
        )
        out.append("")
    for name, body in sections.items():
        out.append(f"<!-- section:{name} -->\n{body}\n<!-- /section:{name} -->")
        out.append("")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(out))


def _write_outputs(
    args,
    meta: Dict[str, Any],
    ledger: Ledger,
    results: List[CaseResult],
    load_info: Optional[Dict[str, Any]],
    recorder,
) -> str:
    section, data = _render(args, meta, ledger, results, load_info, recorder)
    _write_report(args.label, section)
    jsonl = RESULTS_DIR / f"night_monitor_{args.label}.jsonl"
    with jsonl.open("w") as fh:

        def write(record: Dict[str, Any]) -> None:
            fh.write(json.dumps(record, default=str) + "\n")

        write({"type": "meta", **meta})
        for inj in ledger.injections.values():
            write({"type": "injection", **asdict(inj)})
        for record in ledger.scheds.values():
            write({"type": "schedule_event", **asdict(record)})
        for result in results:
            write({"type": "case", **asdict(result)})
        write({"type": "timer_arms", "arms": ledger.timer_arms})
        write({"type": "backlog", "samples": ledger.samples})
        write(
            {
                "type": "summary",
                **data,
                "accounting": _accounting(ledger),
                "unmatched_deliveries": ledger.unmatched_deliveries,
                "load": load_info,
            }
        )
    print(
        f"\nStatus: {data['status']}. Report: {_shown(REPORT_PATH)} (section '{args.label}'); "
        f"events: {_shown(jsonl)}"
    )
    return data["status"]


def _shown(path: Path) -> str:
    try:
        return str(path.relative_to(rt.REPO))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------------------
# selfcheck: offline, no network, nothing written
# --------------------------------------------------------------------------------------


def _selfcheck() -> int:
    rt._prepare_env("local")
    B = _backend()
    failures: List[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(
            f"  {'ok  ' if ok else 'FAIL'} {label}"
            + (f" ({detail})" if detail and not ok else "")
        )
        if not ok:
            failures.append(label)

    print("Patch targets:")
    for owner, name in (
        (B.event_listener, "ResourceEventSource"),
        (B.event_listener, "WeatherEventSource"),
        (B.event_listener, "ODBEventSource"),
        (B.simulationbuilder, "gpp_program_data"),
        (B.EventHandler, "handle"),
        (B.SchedulerQueue, "add_schedule_event"),
        (B.EngineRT, "compute_event_plan"),
        (B.ODBEventHandler, "_on_idle_timeout"),
        (B.ODBEventHandler, "_on_execution_timeout"),
        (B.ODBEventHandler, "on_plan_published"),
        (B.ODBEventHandler, "WAITING_THRESHOLD"),
        (B.ODBEventHandler, "_reference_time"),
        (B.SchedulerIdleTimer, "set"),
        (B.NightTracker, "start_tracking"),
        (B.NightTracker, "_should_trigger_plan"),
        (B.NightTracker, "calculate_night_events"),
        (B.NightMonitor, "get_weather_source"),
        (B.EngineRT, "run"),
    ):
        check(f"{getattr(owner, '__name__', owner)}.{name}", hasattr(owner, name))
    check(
        "EventHandler.handle is not overridden by a handler",
        all(
            "handle" not in cls.__dict__
            for cls in (
                B.ODBEventHandler,
                B.WeatherEventHandler,
                B.ResourceEventHandler,
            )
        ),
    )
    process = B.SchedulerProcess(
        "selfcheck",
        B.SchedulerParameters(
            start=datetime(2026, 10, 14, 8, tzinfo=UTC),
            end=datetime(2026, 10, 28, 8, tzinfo=UTC),
            sites=B.ALL_SITES,
            mode=B.SchedulerModes.OPERATION,
            semester_visibility=False,
            num_nights_to_schedule=1,
        ),
    )
    for name in (
        "scheduler_queue",
        "nightly_timeline_store",
        "night_monitor",
        "engine",
        "_engine_task",
    ):
        check(f"SchedulerProcess().{name}", hasattr(process, name))
    check("SchedulerQueue()._queue", hasattr(B.SchedulerQueue(), "_queue"))

    print("Mock payloads:")
    variants = [
        ("UPDATED READY", {}),
        ("UPDATED ONGOING", {"state": "ONGOING"}),
        ("UPDATED COMPLETED", {"state": "COMPLETED"}),
        ("tight constraints", {"constraints": TIGHT}),
        *[
            (f"UPDATED {s}", {"state": s})
            for s in ("INACTIVE", "UNDEFINED", "UNAPPROVED", "DEFINED")
        ],
        ("CREATED READY", {"edit_type": "CREATED"}),
        ("HARD_DELETE", {"edit_type": "HARD_DELETE"}),
        ("GS instrument", {"site": "GS"}),
    ]
    for label, kwargs in variants:
        try:
            model = B.SchedulerObservationsUpdates.model_validate(
                odb_payload(synthetic_label(), **kwargs)
            )
            value = model.obscalc_update.value
            conditions = B.build_conditions(value)
            site = B.site_key_from_instrument(value.instrument)
            check(
                f"{label}: validates, conditions {conditions.iq}/{conditions.cc.name}, site {site}",
                site == kwargs.get("site", "GN"),
            )
        except Exception as exc:
            check(label, False, f"{type(exc).__name__}: {rt._short(exc)}")
    for label, kwargs in (
        ("no instrument", {"instrument": None}),
        ("reference=None", {"reference": False}),
        ("workflow=None", {"workflow": False}),
        ("value=None", {"value": False}),
    ):
        try:
            B.SchedulerObservationsUpdates.model_validate(
                odb_payload(synthetic_label(), **kwargs)
            )
            check(f"{label}: validates", True)
        except Exception as exc:
            check(
                f"{label}: validates", False, f"{type(exc).__name__}: {rt._short(exc)}"
            )
    bad = odb_payload(synthetic_label())
    bad["obscalcUpdate"]["editType"] = "BOGUS"
    try:
        B.SchedulerObservationsUpdates.model_validate(bad)
        check("malformed frame fails validation", False, "it validated")
    except Exception as exc:
        check(
            "malformed frame fails validation",
            type(exc).__name__ == "ValidationError",
            type(exc).__name__,
        )

    weather = B.WeatherEventHandler(B.SchedulerQueue())
    for site, iq, cc, should_parse in (
        ("GN", 0.7, 0.5, True),
        ("GS", 0.2, 0.7, True),
        ("GN", 0.7, 0.33, False),
        ("GX", 0.7, 0.5, False),
    ):
        try:
            weather.parse_weather_change(weather_payload(site, iq, cc))
            check(f"weather {site} iq={iq} cc={cc} parses", should_parse)
        except Exception as exc:
            check(
                f"weather {site} iq={iq} cc={cc} is rejected ({type(exc).__name__})",
                not should_parse,
            )

    print("Night events and catalog:")
    # astropy would otherwise fetch fresh Earth-rotation tables over the network; the bundled
    # ones are plenty to check that the night events compute.
    from astropy.utils import iers

    iers.conf.auto_download = False
    try:
        events = _night_events(B, datetime(2026, 10, 14, 8, tzinfo=UTC))
        sim_now = _simulated_now(events, "overlap")
        descriptions = [
            events["GS"][NIGHT_INDEX["evening"]].description,
            events["GN"][NIGHT_INDEX["morning"]].description,
        ]
        check(
            f"night events computed offline; simulated now {sim_now:%H:%M} UT",
            all(
                B.NightTracker._should_trigger_plan(SimpleNamespace(description=d))
                for d in descriptions
            ),
        )
        check(
            "sunset does not trigger a plan",
            not B.NightTracker._should_trigger_plan(
                events["GN"][NIGHT_INDEX["sunset"]]
            ),
        )
    except Exception as exc:
        check(
            "night events computed offline",
            False,
            f"{type(exc).__name__}: {rt._short(exc)}",
        )
    names = [c.name for c in catalog()]
    check(f"{len(names)} cases, unique names", len(names) == len(set(names)))

    print(
        "Monitor pipeline (real listener, consumer and handlers; mock feeds; no engine):"
    )
    try:
        failures.extend(asyncio.run(_selfcheck_pipeline(B)))
    except Exception as exc:
        check("pipeline smoke", False, f"{type(exc).__name__}: {rt._short(exc)}")

    print(f"\nselfcheck: {'OK' if not failures else f'{len(failures)} failure(s)'}")
    return 0 if not failures else 1


async def _selfcheck_pipeline(B: SimpleNamespace) -> List[str]:
    failures: List[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(
            f"  {'ok  ' if ok else 'FAIL'} {label}"
            + (f" ({detail})" if detail and not ok else "")
        )
        if not ok:
            failures.append(label)

    ledger = Ledger()
    mocks = MockSources(B, ledger, 0.2, 0.5, resource_subscription=True)
    patches = rt._Patches()
    _install_hooks(patches, B, ledger, mocks, "quiet")
    rt._quiet_scheduler_logs("CRITICAL")
    logging.getLogger("stamina").setLevel(logging.CRITICAL)
    event_queue: asyncio.Queue = asyncio.Queue()
    shutdown = asyncio.Event()
    sched_queue = B.SchedulerQueue()
    store = B.NightlyTimelineStore()
    tasks: List[asyncio.Task] = []
    try:
        listener = B.EventListener(None, event_queue, shutdown)
        consumer = B.EventConsumer(event_queue, shutdown, sched_queue, store)
        check(
            "listener built its sources from the mocks",
            [type(s).__name__ for s in listener._sources]
            == [
                "MockResourceEventSource",
                "MockWeatherEventSource",
                "MockODBEventSource",
            ],
        )
        tasks = [
            asyncio.create_task(listener.listen()),
            asyncio.create_task(consumer.consume()),
        ]

        async def quiet(timeout: float = 10.0) -> None:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if (
                    not any(f.pending() for f in mocks.feeds())
                    and not event_queue._unfinished_tasks
                    and not ledger.handling
                ):
                    await asyncio.sleep(0.2)
                    if not event_queue._unfinished_tasks and not ledger.handling:
                        return
                await asyncio.sleep(0.05)

        deadline = time.monotonic() + 10
        while (
            not all(f.connected for f in mocks.feeds()) and time.monotonic() < deadline
        ):
            await asyncio.sleep(0.05)
        check(
            "all three mock subscriptions connected",
            all(f.connected for f in mocks.feeds()),
        )

        def send(feed: MockFeed, source: str, payload: dict, kind: str) -> Injection:
            inj = ledger.new_injection("selfcheck", source, kind)
            feed.emit(payload, inj)
            return inj

        w_ok = send(
            mocks.weather, "weather", weather_payload("GN", 0.7, 0.5), "weather GN"
        )
        w_bad = send(
            mocks.weather, "weather", weather_payload("GN", 0.7, 0.33), "weather bad cc"
        )
        o_none = send(
            mocks.odb, "odb", odb_payload(synthetic_label()), "UPDATED READY, no plan"
        )
        o_err = send(
            mocks.odb, "odb", odb_payload(synthetic_label(), value=False), "value=None"
        )
        r_err = send(
            mocks.resource, "resource", resource_payload("broken"), "resource broken"
        )
        await quiet()
        check(
            "weather update -> one schedule event, attributed to it",
            len(w_ok.sched_ids) == 1
            and ledger.scheds[w_ok.sched_ids[0]].origin == "handler",
        )
        check(
            "engine state follows the weather feed",
            mocks.weather_state["GN"]["imageQuality"] == 0.7,
        )
        check(
            "bad cloud cover -> handler error",
            bool(w_bad.error) and not w_bad.sched_ids,
            str(w_bad.error),
        )
        check(
            "ODB update with no plan -> handled, nothing sent",
            o_none.t_handle_end is not None
            and not o_none.error
            and not o_none.sched_ids,
        )
        check("ODB value=None -> handler error", bool(o_err.error), str(o_err.error))
        check(
            "resource unknown status -> handler error",
            bool(r_err.error),
            str(r_err.error),
        )
        check(
            "processing time recorded",
            all(
                i.t_received is not None and i.t_handle_start is not None
                for i in (w_ok, o_none)
            ),
        )

        before = mocks.odb.connections
        bad = odb_payload(synthetic_label())
        bad["obscalcUpdate"]["editType"] = "BOGUS"
        o_bad = send(mocks.odb, "odb", bad, "malformed")
        deadline = time.monotonic() + 30
        while (
            not (mocks.odb.connected and mocks.odb.connections > before)
            and time.monotonic() < deadline
        ):
            await asyncio.sleep(0.05)
        check(
            "malformed ODB frame -> lost, subscription reconnects (stamina)",
            o_bad.lost == "validation" and mocks.odb.connections > before,
            f"lost={o_bad.lost}",
        )
        o_after = send(
            mocks.odb, "odb", odb_payload(synthetic_label()), "after reconnect"
        )
        await quiet()
        check("event after the reconnect is handled", o_after.t_handle_end is not None)

        before = mocks.weather.connections
        mocks.weather.fault("closed")
        deadline = time.monotonic() + 5
        while mocks.weather.connected and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        w_lost = send(
            mocks.weather, "weather", weather_payload("GS", 0.7, 0.5), "while down"
        )
        check(
            "event sent while the socket is down -> lost",
            w_lost.lost == "disconnected",
            str(w_lost.lost),
        )
        deadline = time.monotonic() + 30
        while (
            not (mocks.weather.connected and mocks.weather.connections > before)
            and time.monotonic() < deadline
        ):
            await asyncio.sleep(0.05)
        check(
            "weather subscription reconnects after a dropped socket",
            mocks.weather.connections > before,
        )
        check(
            "schedule events reached the real SchedulerQueue",
            sched_queue._queue.qsize() == len(ledger.scheds),
        )
    finally:
        shutdown.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        patches.undo()
    return failures


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    # Checked in main rather than required here, so `cases --list` needs neither.
    parser.add_argument("--visibility", choices=("local", "sight"), help="Required.")
    parser.add_argument(
        "--label", help="Required. Report section name, e.g. nm-cases, nm-load."
    )
    parser.add_argument(
        "--source",
        choices=("clones", "odb"),
        default="clones",
        help="clones (default): programs active on the night, fetched once and cloned to "
        "--size per site. odb: no clones; every plan fetches from the ODB itself.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=250,
        help="Observations per site with clones (default 250).",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=rt.DEFAULT_CAP,
        help=f"Clone numbering cap; keep it equal to seed-sight's (default {rt.DEFAULT_CAP}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=rt.DEFAULT_SEED,
        help="Random seed for clone positions and the load stream.",
    )
    parser.add_argument(
        "--programs", help="Comma-separated reference labels or ODB ids to fetch."
    )
    parser.add_argument(
        "--allow-nonsidereal",
        action="store_true",
        help="Keep programs with non-sidereal targets (local visibility then calls JPL Horizons).",
    )
    parser.add_argument(
        "--night", help="UT date of the 08:00 anchor (YYYY-MM-DD). Defaults to tonight."
    )
    parser.add_argument(
        "--vis-end",
        metavar="semester|realtime-default|YYYY-MM-DD",
        default="realtime-default",
        help="How far the engine reads visibility (default realtime-default: 15 nights).",
    )
    parser.add_argument(
        "--sim-now",
        default="overlap",
        help="Simulated clock start: 'overlap' (default, both sites dark) or hours after the "
        "first evening twilight.",
    )
    parser.add_argument(
        "--timer-mode",
        choices=("quiet", "production"),
        default="quiet",
        help="quiet (default): the 10 min idle timer becomes 6 h, so only the timer cases "
        "replan on a timer. production: keep 10 min.",
    )
    parser.add_argument(
        "--timer-wait",
        type=float,
        default=10.0,
        help="Seconds a timer case's own timer is shortened to (default 10).",
    )
    parser.add_argument(
        "--case-timeout",
        type=float,
        default=180.0,
        help="Seconds a case (or the seed) may take to settle (default 180).",
    )
    parser.add_argument(
        "--iq",
        type=float,
        default=0.2,
        help="Initial image quality, both sites (default 0.2).",
    )
    parser.add_argument(
        "--cc",
        type=float,
        default=0.5,
        help="Initial cloud cover, both sites (default 0.5).",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Scheduler log level during the run (default WARNING).",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    sub.add_parser(
        "selfcheck",
        help="Offline: patch targets, payloads and the monitor half of the pipeline.",
    )

    cases = sub.add_parser(
        "cases", help="Run the case catalog and check every outcome."
    )
    cases.add_argument("--list", action="store_true", help="List the cases and exit.")
    cases.add_argument(
        "--cases", help="Comma-separated case names or groups (default: all)."
    )
    _add_run_args(cases)

    load = sub.add_parser(
        "load", help="Sustained mixed stream: backlog, latency under load, loss."
    )
    load.add_argument(
        "--rate",
        type=float,
        default=0.2,
        help="Events per second, Poisson (default 0.2).",
    )
    load.add_argument(
        "--duration",
        type=float,
        default=600.0,
        help="Seconds of sending (default 600).",
    )
    load.add_argument("--count", type=int, help="Stop after this many events.")
    load.add_argument(
        "--burst", type=int, help="Send this many events at once instead of a stream."
    )
    load.add_argument(
        "--mix",
        default=DEFAULT_MIX,
        help=f"Event kinds and weights (default {DEFAULT_MIX}).",
    )
    load.add_argument(
        "--drop-every", type=int, help="Drop the ODB socket after every N ODB events."
    )
    load.add_argument(
        "--drain-timeout",
        type=float,
        default=3600.0,
        help="Seconds to wait for the backlog to clear after sending (default 3600).",
    )
    _add_run_args(load)

    args = parser.parse_args()
    if args.mode == "selfcheck":
        return _selfcheck()
    if args.mode == "cases" and args.list:
        for case in catalog():
            flags = [
                f
                for f, on in (
                    ("known bug", case.known_bug),
                    ("synthetic", case.synthetic),
                    ("record only", case.expect.record_only),
                )
                if on
            ]
            print(
                f"{case.group:9} {case.name:38} {case.summary}"
                + (f"  [{', '.join(flags)}]" if flags else "")
            )
        return 0

    if not args.visibility or not args.label:
        sys.exit(f"{args.mode} needs --visibility and --label.")
    if not rt._LABEL_ARG.match(args.label):
        sys.exit("--label may only hold letters, digits, '.', '_' and '-'.")
    if args.source == "clones" and not 0 < args.size <= args.cap:
        sys.exit(f"--size must be between 1 and --cap ({args.cap}).")
    if args.mode == "load":
        if args.rate <= 0 and not args.burst:
            sys.exit("--rate must be positive.")
        _parse_mix(args.mix)
    if args.visibility == "sight":
        rt._require_database_url(f"{args.mode} --visibility sight")
    rt._prepare_env(args.visibility)
    return asyncio.run(_measure(args))


if __name__ == "__main__":
    sys.exit(main())
