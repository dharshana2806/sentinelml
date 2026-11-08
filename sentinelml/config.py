"""Central configuration objects (dataclasses) so experiments are reproducible."""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class SimConfig:
    """Configuration of one simulated deployment stream.

    One *window* is the unit of time (think "one hour of production traffic").
    """
    window_size: int = 100          # rows per monitoring window
    n_windows: int = 600            # stream length
    n_events: int = 10              # injected failure events per stream
    warmup: int = 40                # windows with no events at the start
    label_delay: int = 2            # ground truth for window t arrives at t + label_delay
    horizon: int = 6                # prediction horizon H (windows)
    smooth: int = 3                 # causal smoothing of loss when defining "degraded"
    degrade_k: float = 4.0          # threshold = mean + max(k*std, rel*mean) of healthy smoothed loss
    degrade_rel: float = 0.20
    degrade_abs: float = 0.04       # minimum absolute loss increase that counts as degradation
    min_event_len: int = 30
    max_event_len: int = 44
    p_sudden: float = 0.3           # fraction of events that are sudden (vs gradual ramp)

    def to_dict(self) -> dict:
        return asdict(self)


FAILURE_TYPES = [
    "covariate_shift", "concept_drift", "missing_values", "feature_corruption",
    "label_shift", "pipeline_mismatch", "outliers",
]
