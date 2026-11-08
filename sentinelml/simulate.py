"""Deployment simulator: trains a model, streams windows, injects failures, labels degradation."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import SimConfig
from .datasets import TaskData, load_dataset
from .drift import RefProfile
from .failures import FailureEvent, schedule_events, apply_event
from .model import TaskModel
from .monitor import window_signals


@dataclass
class Context:
    """Everything produced at 'deploy time'."""
    td: TaskData
    model: TaskModel
    prev_model: TaskModel          # an older registered version (used by the rollback candidate)
    profile: RefProfile
    thr: float                     # degradation threshold on smoothed loss
    thr_lo: float                  # exit threshold (hysteresis)
    base_loss: float
    n_classes: int
    cfg: SimConfig


def smooth_loss(loss: np.ndarray, k: int) -> np.ndarray:
    return pd.Series(loss).rolling(k, min_periods=1).mean().to_numpy()


def hysteresis_state(sm: np.ndarray, hi: float, lo: float) -> np.ndarray:
    """Degraded from the first time sm > hi until sm < lo (prevents flicker around the threshold)."""
    st = np.zeros(len(sm), bool)
    cur = False
    for i, v in enumerate(sm):
        if not cur and v > hi:
            cur = True
        elif cur and v < lo:
            cur = False
        st[i] = cur
    return st


def _draw(X: pd.DataFrame, y: np.ndarray, n: int, rng) -> tuple:
    idx = rng.choice(len(X), n, replace=len(X) < n)
    return X.iloc[idx].reset_index(drop=True), y[idx]


def build_context(name: str, seed: int, cfg: SimConfig) -> Context:
    td = load_dataset(name, seed)
    model = TaskModel(td.task, seed=seed).fit(td.X_train, td.y_train)
    sub = np.random.default_rng(seed + 1).choice(len(td.X_train), int(0.6 * len(td.X_train)), replace=False)
    prev = TaskModel(td.task, n_estimators=30, seed=seed + 7, max_depth=8).fit(td.X_train.iloc[sub], td.y_train[sub])
    out_ref = model.outputs(td.X_ref)
    profile = RefProfile.build(td.X_ref, model.feature_importances_, out_ref, seed=seed)
    # degradation threshold from a clean (event-free) simulation of production traffic: windows drawn
    # from the production pool, with a different RNG stream than the evaluation stream
    rng = np.random.default_rng(seed + 99)
    losses = []
    for _ in range(200):
        Xw, yw = _draw(td.X_pool, td.y_pool, cfg.window_size, rng)
        losses.append(model.loss(yw, model.predict(Xw)))
    sm = smooth_loss(np.array(losses), cfg.smooth)[cfg.smooth:]
    thr = float(sm.mean() + max(cfg.degrade_k * sm.std(), cfg.degrade_rel * sm.mean(), cfg.degrade_abs))
    n_classes = int(len(np.unique(td.y_train))) if td.is_classification else 0
    base = float(sm.mean())
    return Context(td, model, prev, profile, thr, base + 0.5 * (thr - base), base, n_classes, cfg)


@dataclass
class StreamResult:
    name: str
    seed: int
    signals: pd.DataFrame
    loss: np.ndarray
    degraded: np.ndarray
    events: list
    windows: list = field(default_factory=list)      # list[(X DataFrame, y)] (kept for RCA / recovery)

    @property
    def onsets(self) -> np.ndarray:
        d = self.degraded.astype(int)
        return np.where((d == 1) & (np.r_[0, d[:-1]] == 0))[0]


def simulate_stream(ctx: Context, seed: int, keep_windows: bool = True, types: list | None = None,
                    event_filter=None) -> StreamResult:
    cfg, td = ctx.cfg, ctx.td
    rng = np.random.default_rng(seed)
    events = schedule_events(cfg.n_windows, cfg.n_events, rng, td.feature_names, ctx.profile.importance,
                             td.task, cfg.warmup, cfg.min_event_len, cfg.max_event_len, cfg.p_sudden, types)
    if event_filter:
        events = [e for e in events if event_filter(e)]
    rows, losses, wins = [], [], []
    for t in range(cfg.n_windows):
        Xw, yw = _draw(td.X_pool, td.y_pool, cfg.window_size, rng)
        eid, etype, eint = -1, "none", 0.0
        for e in events:
            if e.active(t):
                Xw, yw = apply_event(e, Xw, yw, t, ctx.profile, np.random.default_rng([seed, e.event_id, t]),
                                     td.task, ctx.n_classes)
                eid, etype, eint = e.event_id, e.ftype, e.severity * e.intensity(t)
        out = ctx.model.outputs(Xw)
        loss = ctx.model.loss(yw, out["pred"])
        sig = window_signals(ctx.profile, Xw, out)
        sig.update(loss=loss, event_id=eid, event_type=etype, event_intensity=eint)
        rows.append(sig); losses.append(loss)
        if keep_windows:
            wins.append((Xw, yw))
    df = pd.DataFrame(rows)
    df.index.name = "t"
    loss = np.array(losses)
    # labels arrive with a delay -> the only performance signal usable online is the lagged one
    df["loss_lag"] = pd.Series(loss).shift(cfg.label_delay)
    degraded = hysteresis_state(smooth_loss(loss, cfg.smooth), ctx.thr, ctx.thr_lo)
    return StreamResult(td.name, seed, df, loss, degraded, events, wins)


def calibrate_threshold(model: TaskModel, X: pd.DataFrame, y: np.ndarray, cfg: SimConfig, seed: int = 0) -> tuple:
    """(base_loss, degrade_threshold) from clean windows of (X, y). Used by the serving API."""
    rng = np.random.default_rng(seed)
    w = max(10, min(cfg.window_size, len(X) // 2))
    losses = np.array([model.loss(*(lambda a: (a[1], model.predict(a[0])))(_draw(X, y, w, rng))) for _ in range(100)])
    sm = smooth_loss(losses, cfg.smooth)[cfg.smooth:]
    base = float(sm.mean())
    return base, float(base + max(cfg.degrade_k * sm.std(), cfg.degrade_rel * base, cfg.degrade_abs))
