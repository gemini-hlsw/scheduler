from dataclasses import dataclass
from typing import final, TypeAlias, Dict, Tuple

Summary: TypeAlias = Dict[str, Tuple[str, float]]
MetricPerBand: TypeAlias = Dict[str, float]


__all__ = ['RunSummary']

@final
@dataclass
class RunSummary:
    summary: Summary
    metrics_per_band: MetricPerBand