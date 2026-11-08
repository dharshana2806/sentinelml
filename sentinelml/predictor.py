"""Predictive failure intelligence: will the model *start* degrading within the next H windows?

Formulation (binary, onset prediction): for every window t at which the model is currently healthy,
    y_t = 1  iff  the model enters a degraded state in (t, t+H].
Features: causal rolling features of drift / quality / uncertainty / lagged-performance signals
(`monitor.build_features`). Evaluation is strictly chronological with purge gaps of H windows so that
label look-ahead never crosses a split boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import SimConfig
from .monitor import build_features
from .simulate import StreamResult

SPLITS = (0.5, 0.65)   # train < 50 % of the stream, validation 50-65 %, test > 65 %


def episodes(degraded: np.ndarray) -> list:
    """List of (onset, end_exclusive) degraded episodes."""
    d = degraded.astype(int)
    on = np.where((d == 1) & (np.r_[0, d[:-1]] == 0))[0]
    off = np.where((d == 0) & (np.r_[0, d[:-1]] == 1))[0]
    return [(int(o), int(off[off > o][0]) if (off > o).any() else len(d)) for o in on]


def build_table(results: list, cfg: SimConfig, temporal: bool = True, drop_groups: tuple = ()) -> pd.DataFrame:
    """Pooled supervised table over streams. Columns: features + meta (stream, dataset, t, y, healthy, seg, ...)."""
    H, T = cfg.horizon, cfg.n_windows
    frames = []
    for sid, r in enumerate(results):
        F = build_features(r.signals, temporal=temporal, drop_groups=drop_groups)
        deg = r.degraded
        fut = np.array([deg[t + 1: t + H + 1].any() if t + H < T else np.nan for t in range(T)], dtype=float)
        meta = pd.DataFrame({"stream": f"{r.name}#{r.seed}", "dataset": r.name, "t": np.arange(T), "y": fut,
                             "healthy": ~deg, "degraded": deg,
                             "event_type": r.signals["event_type"].to_numpy(),
                             "event_intensity": r.signals["event_intensity"].to_numpy()})
        seg = np.where(meta.t < SPLITS[0] * T - H, "train",
                       np.where((meta.t >= SPLITS[0] * T) & (meta.t < SPLITS[1] * T - H), "val",
                                np.where(meta.t >= SPLITS[1] * T, "test", "purge")))
        meta["seg"] = seg
        # attach shape/severity of the event responsible for each upcoming onset
        sev = np.full(T, np.nan); shape = np.array([""] * T, dtype=object)
        for o, _ in episodes(deg):
            ev = [e for e in r.events if e.start <= o <= e.end + 3]
            if ev:
                e = ev[-1]
                lo = max(0, o - H)
                sev[lo:o] = e.severity; shape[lo:o] = e.shape
        meta["onset_severity"] = sev; meta["onset_shape"] = shape
        frames.append(pd.concat([meta, F.reset_index(drop=True)], axis=1))
    return pd.concat(frames, ignore_index=True)


def feature_columns(table: pd.DataFrame) -> list:
    meta = {"stream", "dataset", "t", "y", "healthy", "degraded", "event_type", "event_intensity", "seg",
            "onset_severity", "onset_shape"}
    return [c for c in table.columns if c not in meta]


# ------------------------------------------------------------------------------------------------
def make_model(name: str, seed: int = 0):
    if name == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000))
    if name == "rf":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(200, min_samples_leaf=5, class_weight="balanced_subsample",
                                                    n_jobs=1, random_state=seed))
    if name == "hgb":
        return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200,
                                              l2_regularization=1.0, random_state=seed)
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, n_jobs=1, random_state=seed, verbosity=0)
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, n_jobs=1, random_state=seed,
                              verbose=-1)
    raise ValueError(name)


MODEL_NAMES = ["logreg", "rf", "hgb", "xgb", "lgbm"]


def best_f1_threshold(y: np.ndarray, s: np.ndarray) -> float:
    if y.sum() == 0:
        return 0.5
    cand = np.unique(np.quantile(s, np.linspace(0.5, 0.995, 100)))
    best, bt = -1.0, float(cand[-1])
    for t in cand:
        p, r, f, _ = precision_recall_fscore_support(y, s >= t, average="binary", zero_division=0)
        if f > best:
            best, bt = f, float(t)
    return bt


@dataclass
class FailurePredictor:
    """Trained failure-onset predictor with a validation-tuned alert threshold."""
    name: str
    columns: list
    model: object
    threshold: float = 0.5
    calibrator: object | None = None
    horizon: int = 6

    def fit(self, tr: pd.DataFrame, va: pd.DataFrame | None = None) -> "FailurePredictor":
        tr = tr[(tr.healthy) & tr.y.notna()]
        self.model.fit(tr[self.columns], tr.y.astype(int))
        if va is not None:
            va = va[(va.healthy) & va.y.notna()]
            s = self.model.predict_proba(va[self.columns])[:, 1]
            self.threshold = best_f1_threshold(va.y.to_numpy(int), s)
            if va.y.sum() >= 5 and (1 - va.y).sum() >= 5:
                self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s, va.y)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.columns])[:, 1]

    def calibrated(self, s: np.ndarray) -> np.ndarray:
        return self.calibrator.predict(s) if self.calibrator is not None else s

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FailurePredictor":
        return joblib.load(path)


# ------------------------------------------------------------------------------------------------
def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    idx = np.minimum((p * bins).astype(int), bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def window_metrics(y: np.ndarray, s: np.ndarray, alert: np.ndarray, p_cal: np.ndarray | None = None) -> dict:
    y = y.astype(int)
    out = {"n": len(y), "prevalence": float(y.mean()) if len(y) else float("nan")}
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return {**out, **{k: float("nan") for k in ("pr_auc", "roc_auc", "precision", "recall", "f1",
                                                    "false_alarm_per_100", "brier", "ece")}}
    p, r, f, _ = precision_recall_fscore_support(y, alert, average="binary", zero_division=0)
    out.update(pr_auc=float(average_precision_score(y, s)), roc_auc=float(roc_auc_score(y, s)),
               precision=float(p), recall=float(r), f1=float(f),
               false_alarm_per_100=float(100 * (alert & (y == 0)).sum() / max((y == 0).sum(), 1)))
    if p_cal is not None:
        out.update(brier=float(brier_score_loss(y, p_cal)), ece=ece(y, p_cal))
    return out


def episode_metrics(tbl: pd.DataFrame, alert_col: str, H: int) -> dict:
    """Warning lead time, detection rate and detection delay over degradation episodes in the test region.

    An episode with onset o is *warned* if an alert fired at some healthy t in [o-H, o-1].
    Lead time = o - (first such t). For unwarned episodes, detection delay = first alert at t >= o
    (alert rule applied on all windows) minus o; episodes never alerted are counted as missed.
    """
    leads, delays, n, warned, missed = [], [], 0, 0, 0
    for _, g in tbl.groupby("stream"):
        g = g[g.seg == "test"].sort_values("t")          # positional index 0 == first test window
        deg = g.degraded.to_numpy(); al = g[alert_col].to_numpy().astype(bool)
        for o, e in episodes(deg):
            if o < H:                                      # onset needs a full look-back inside the test region
                continue
            n += 1
            pre = np.where(al[o - H: o])[0]
            if len(pre):
                warned += 1; leads.append(H - pre[0])
            else:
                post = np.where(al[o:e])[0]
                if len(post):
                    delays.append(int(post[0]))
                else:
                    missed += 1
    return {"episodes": n, "warned_rate": warned / n if n else float("nan"),
            "mean_lead": float(np.mean(leads)) if leads else float("nan"),
            "mean_delay_unwarned": float(np.mean(delays)) if delays else float("nan"),
            "missed_rate": missed / n if n else float("nan")}


def bootstrap_ci(tbl: pd.DataFrame, fn, n_boot: int = 200, seed: int = 0) -> tuple:
    """Percentile 95 % CI of fn(sub_table) by resampling whole streams."""
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in tbl.groupby("stream")}
    keys = list(groups)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys), replace=True)
        sub = pd.concat([groups[keys[i]] for i in pick])
        v = fn(sub)
        if v == v:
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
