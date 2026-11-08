"""Failure-injection framework: reproducible, ground-truth-labelled production failures.

Every FailureEvent has type, severity, start, duration, shape (sudden | gradual ramp) and affected
features, so that detection, lead time and root-cause accuracy can be scored against ground truth.

Why each scenario is realistic
  covariate_shift    upstream population changes (new customer segment, seasonality)
  concept_drift      relationship P(y|x) changes (fraud tactics, consumer behaviour)
  missing_values     broken upstream join / sensor outage -> NaNs
  feature_corruption sensor noise / degraded data source
  label_shift        class prior changes (class imbalance appears) - often harmless for accuracy!
  pipeline_mismatch  unit/scale change in a feature (cm -> mm) after a pipeline release
  outliers           bursts of extreme values (logging bugs, sentinel values)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from .config import FAILURE_TYPES

FEATURE_TYPES = {"covariate_shift", "missing_values", "feature_corruption", "pipeline_mismatch", "outliers"}


@dataclass
class FailureEvent:
    event_id: int
    ftype: str
    severity: float                 # 0..1
    start: int
    duration: int
    shape: str                      # "sudden" | "gradual"
    ramp: int                       # windows to reach full intensity when gradual
    features: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def end(self) -> int:
        return self.start + self.duration

    def active(self, t: int) -> bool:
        return self.start <= t < self.end

    def intensity(self, t: int) -> float:
        if not self.active(t):
            return 0.0
        if self.shape == "sudden":
            return 1.0
        return min(1.0, (t - self.start + 1) / max(self.ramp, 1))

    def to_dict(self) -> dict:
        return asdict(self)


def schedule_events(n_windows: int, n_events: int, rng: np.random.Generator, features: list,
                    importance: np.ndarray, task: str, warmup: int = 40, min_len: int = 30,
                    max_len: int = 44, p_sudden: float = 0.3, types: list | None = None) -> list:
    """Place events in equal slots (no overlap, >=12 windows of recovery between events)."""
    types = types or FAILURE_TYPES
    slot = (n_windows - warmup) // n_events
    order = np.argsort(-importance)
    top = [features[i] for i in order[: max(3, int(0.6 * len(features)))]]
    events = []
    for k in range(n_events):
        dur = int(rng.integers(min_len, min(max_len, slot - 13) + 1))
        start = warmup + k * slot + int(rng.integers(0, max(1, slot - dur - 12)))
        ftype = types[int(rng.integers(len(types)))]
        shape = "sudden" if rng.random() < p_sudden else "gradual"
        sev = float(rng.uniform(0.4, 1.0))
        feats: list = []
        meta: dict = {}
        if ftype in FEATURE_TYPES:
            nf = int(rng.integers(1, max(3, round(0.25 * len(features))) + 1))
            pool = top if rng.random() < 0.7 else features
            feats = [str(f) for f in rng.choice(pool, size=min(nf, len(pool)), replace=False)]
        if ftype == "label_shift":
            meta["target_class"] = int(rng.integers(0, 2))
        if ftype == "concept_drift":
            meta["concept_feature"] = str(top[0])
        events.append(FailureEvent(k, ftype, sev, start, dur, shape, int(rng.integers(8, 17)), feats, meta))
    return events


def apply_event(ev: FailureEvent, X: pd.DataFrame, y: np.ndarray, t: int, profile,
                rng: np.random.Generator, task: str, n_classes: int = 2):
    """Return the corrupted (X, y) for window t. Inputs are not modified."""
    s = ev.severity * ev.intensity(t)
    if s <= 0:
        return X, y
    X = X.copy(); y = np.array(y, copy=True)
    n = len(X)
    if ev.ftype == "covariate_shift":
        for f in ev.features:
            X[f] = X[f] + 3.0 * s * profile.std[f]
    elif ev.ftype == "missing_values":
        for f in ev.features:
            X.loc[rng.random(n) < 0.8 * s, f] = np.nan
    elif ev.ftype == "feature_corruption":
        for f in ev.features:
            X[f] = X[f] + rng.normal(0, 3.0 * s * profile.std[f], n)
    elif ev.ftype == "pipeline_mismatch":
        for f in ev.features:
            X[f] = X[f] * 10 ** (2.0 * s)
    elif ev.ftype == "outliers":
        for f in ev.features:
            m = rng.random(n) < 0.15 * s
            X.loc[m, f] = profile.mean[f] + rng.choice([-1, 1], m.sum()) * 8 * profile.std[f]
    elif ev.ftype == "concept_drift":
        p = 0.6 * s
        if task == "regression":
            f = ev.meta["concept_feature"]
            z = (X[f].to_numpy() - profile.mean[f]) / profile.std[f]
            y = y + s * 2.0 * np.std(y) * np.nan_to_num(z)
        else:
            m = rng.random(n) < p
            y[m] = (y[m] + 1) % n_classes          # systematic relabelling of a fraction of the space
    elif ev.ftype == "label_shift":
        a = 0.9 * s
        if task == "regression":
            z = (y - y.mean()) / (y.std() or 1.0)
            w = np.exp(a * 3 * z)
        else:
            tgt = ev.meta.get("target_class", 0) % n_classes
            w = (1 - a) + a * (y == tgt) * n_classes
        w = w / w.sum()
        idx = rng.choice(n, n, replace=True, p=w)
        X = X.iloc[idx].reset_index(drop=True); y = y[idx]
    return X, y
