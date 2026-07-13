# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, Optional, Tuple

from astropy.time import Time
from lucupy.minimodel import Site, VariantSnapshot

from scheduler.core.builder import Blueprints, SimulationBuilder
from scheduler.core.builder.modes import dispatch_with
from scheduler.core.components.ranker import DefaultRanker
from scheduler.core.events.queue.events import Event
from scheduler.core.plans import Plans
from scheduler.core.scp.scp import SCP
from scheduler.core.sources import Sources
from scheduler.engine.params import BuildParameters, SchedulerParameters
from scheduler.engine.plan_computation import compute_event_plans
from scheduler.services import logger_factory

__all__ = ['ComputePayload', 'PlanWorker', 'worker_build', 'worker_compute']

_logger = logger_factory.create_logger(__name__)


@dataclass(frozen=True)
class ComputePayload:
    """Everything worker_compute needs for one plan, in picklable form."""
    event: Event
    sites: FrozenSet[Site]
    night_times: Optional[Dict[Site, Tuple[Optional[Time], Optional[Time]]]]
    variants: Dict[Site, VariantSnapshot] = field(default_factory=dict)


def _build_scp(params: SchedulerParameters, build_params: BuildParameters) -> SCP:
    """Synchronous SCP construction (mirrors EngineRT.build).

    The collector build is async upstream, so it runs under its own event loop
    here — the worker process has no loop of its own.
    """
    builder = dispatch_with(Sources(), None)
    if not isinstance(builder, SimulationBuilder):
        raise RuntimeError("Builder must be Simulation to build the RT pipeline.")

    night_times = build_params.get_night_times()
    vis_start = build_params.visibility_start or params.start
    vis_end = build_params.visibility_end or params.end_vis
    programs_list = build_params.program_list or params.programs_list

    collector = asyncio.run(builder.async_build_collector(
        start=vis_start,
        end=vis_end,
        num_of_nights=params.num_nights_to_schedule,
        sites=params.sites,
        semesters=params.semesters,
        blueprint=Blueprints.collector,
        night_times=night_times,
        program_list=programs_list,
    ))

    selector = builder.build_selector(collector=collector,
                                      num_nights_to_schedule=params.num_nights_to_schedule,
                                      blueprint=Blueprints.selector)
    optimizer = builder.build_optimizer(Blueprints.optimizer)
    ranker = DefaultRanker(collector,
                           params.night_indices,
                           params.sites,
                           params=params.ranker_parameters)

    return SCP(collector, selector, optimizer, ranker)


class PlanWorker:
    """The worker process's warm pipeline: builds the SCP once, computes many.

    The build function is injected so tests exercise the caching/compute logic
    without constructing a real pipeline.
    """

    def __init__(self, build_scp: Callable[[SchedulerParameters, BuildParameters], SCP] = _build_scp) -> None:
        self._build_scp = build_scp
        self._scp: Optional[SCP] = None

    @property
    def is_built(self) -> bool:
        return self._scp is not None

    def build(self, params: SchedulerParameters, build_params: BuildParameters) -> bool:
        """Build (or rebuild) the cached SCP. Returns True on success."""
        _logger.info("Worker: building SCP...")
        self._scp = self._build_scp(params, build_params)
        _logger.info("Worker: SCP built and cached.")
        return True

    def compute(self, payload: ComputePayload) -> Plans:
        """Compute plans for one event against the cached SCP.

        Raises:
            RuntimeError: if build has not run in this worker process.
        """
        if self._scp is None:
            raise RuntimeError("compute called before build: the worker has no SCP.")

        for site, variant in payload.variants.items():
            self._scp.selector.update_site_variant(site, variant)

        return compute_event_plans(self._scp, payload.sites, payload.event, payload.night_times)


# One PlanWorker per worker process. ProcessPoolExecutor can only invoke
# top-level functions and the warm SCP must survive across calls, so the
# instance is anchored at module scope; all mutable state lives inside it.
_worker = PlanWorker()


def worker_build(params: SchedulerParameters, build_params: BuildParameters) -> bool:
    return _worker.build(params, build_params)


def worker_compute(payload: ComputePayload) -> Plans:
    return _worker.compute(payload)
