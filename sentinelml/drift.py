"""Drift statistics and data-quality checks.

Statistical notes / limitations (see docs/METHODOLOGY.md):
* PSI  - histogram based, depends on binning; rule-of-thumb thresholds (0.1 / 0.25) are conventions,
         not statistical tests. Unstable for tiny windows.
* KS   - distribution-free for continuous data, sensitive to location/shape changes near the centre,
         less so in the tails; p-values are meaningless to compare across many windows/features
         (multiple testing) so we use the *statistic* as an effect size.
* Wasserstein - interpretable in feature units (we normalise by the reference std); sensitive to tails.
* JS divergence - symmetric, bounded [0,1] (base 2), histogram based like PSI.
All of them are *univariate*: they miss changes in dependence structure (we add a correlation-shift
signal) and none of them proves that model performance degraded.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

EPS = 1e-4
N_BINS = 10


def _edges(ref: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    qs = np.quantile(ref, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.unique(qs)


def _props(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges, x, side="right")
    c = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    return np.clip(c / max(c.sum(), 1), EPS, None)


def psi_from_props(p_ref: np.ndarray, p_cur: np.ndarray) -> float:
    p_ref = p_ref / p_ref.sum()
    p_cur = p_cur / p_cur.sum()
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def psi(ref: np.ndarray, cur: np.ndarray, n_bins: int = N_BINS) -> float:
    """Population Stability Index of `cur` against `ref` using reference-quantile bins."""
    ref = np.asarray(ref, float); cur = np.asarray(cur, float)
    ref = ref[~np.isnan(ref)]; cur = cur[~np.isnan(cur)]
    if len(ref) < 5 or len(cur) < 5:
        return float("nan")
    e = _edges(ref, n_bins)
    return psi_from_props(_props(ref, e), _props(cur, e))


def ks_statistic(ref: np.ndarray, cur: np.ndarray) -> float:
    ref = np.asarray(ref, float); cur = np.asarray(cur, float)
    ref = ref[~np.isnan(ref)]; cur = cur[~np.isnan(cur)]
    if len(ref) < 5 or len(cur) < 5:
        return float("nan")
    return float(stats.ks_2samp(ref, cur, method="asymp").statistic)


def wasserstein_norm(ref: np.ndarray, cur: np.ndarray) -> float:
    """1-D Wasserstein distance divided by the reference standard deviation."""
    ref = np.asarray(ref, float); cur = np.asarray(cur, float)
    ref = ref[~np.isnan(ref)]; cur = cur[~np.isnan(cur)]
    if len(ref) < 5 or len(cur) < 5:
        return float("nan")
    s = ref.std() or 1.0
    return float(stats.wasserstein_distance(ref, cur) / s)


def js_divergence(ref: np.ndarray, cur: np.ndarray, n_bins: int = N_BINS) -> float:
    ref = np.asarray(ref, float); cur = np.asarray(cur, float)
    ref = ref[~np.isnan(ref)]; cur = cur[~np.isnan(cur)]
    if len(ref) < 5 or len(cur) < 5:
        return float("nan")
    e = _edges(ref, n_bins)
    return float(jensenshannon(_props(ref, e), _props(cur, e), base=2) ** 2)



def ks_wd_fast(ref_sorted: np.ndarray, cur: np.ndarray) -> tuple:
    """Exact two-sample KS statistic and 1-D Wasserstein-1 distance in one pass (vectorised)."""
    c = np.sort(cur)
    allv = np.sort(np.concatenate([ref_sorted, c]))
    fr = np.searchsorted(ref_sorted, allv, side="right") / len(ref_sorted)
    fc = np.searchsorted(c, allv, side="right") / len(c)
    diff = np.abs(fr - fc)
    return float(diff.max()), float(np.sum(diff[:-1] * np.diff(allv)))


def _js_props(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum(); q = q / q.sum(); m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m)))


# ------------------------------------------------------------------------------------------------
@dataclass
class RefProfile:
    """Everything we keep about the reference data so windows can be compared cheaply."""
    feature_names: list
    ref_sorted: dict            # feature -> sorted reference sample (<=500 values)
    edges: dict                 # feature -> histogram edges
    props: dict                 # feature -> reference bin proportions
    mean: dict
    std: dict
    iqr_lo: dict
    iqr_hi: dict
    missing: dict
    corr: np.ndarray
    median: dict
    importance: np.ndarray
    out: dict = field(default_factory=dict)   # reference model-output profile (score/unc/class dist)
    n_ref: int = 0

    @classmethod
    def build(cls, X_ref: pd.DataFrame, importance: np.ndarray, outputs: dict | None = None,
              max_sample: int = 300, seed: int = 0) -> "RefProfile":
        rng = np.random.default_rng(seed)
        feats = list(X_ref.columns)
        d = {k: {} for k in ("ref_sorted", "edges", "props", "mean", "std", "iqr_lo", "iqr_hi", "missing",
                              "median")}
        for f in feats:
            v = X_ref[f].to_numpy(float)
            miss = float(np.isnan(v).mean())
            v = v[~np.isnan(v)]
            sub = v if len(v) <= max_sample else rng.choice(v, max_sample, replace=False)
            e = _edges(v)
            d["ref_sorted"][f] = np.sort(sub)
            d["edges"][f] = e
            d["props"][f] = _props(v, e)
            d["mean"][f] = float(v.mean()); d["std"][f] = float(v.std() or 1.0)
            q1, q3 = np.quantile(v, [0.25, 0.75]); iqr = (q3 - q1) or d["std"][f]
            d["iqr_lo"][f] = float(q1 - 3 * iqr); d["iqr_hi"][f] = float(q3 + 3 * iqr)
            d["missing"][f] = miss
            d["median"][f] = float(np.median(v))
        corr = np.nan_to_num(np.corrcoef(X_ref.fillna(X_ref.median()).to_numpy().T))
        prof = cls(feats, d["ref_sorted"], d["edges"], d["props"], d["mean"], d["std"], d["iqr_lo"],
                   d["iqr_hi"], d["missing"], corr, d["median"], np.asarray(importance, float),
                   n_ref=len(X_ref))
        if outputs is not None:
            prof.out = {
                "score_sorted": np.sort(outputs["score"]),
                "score_edges": _edges(outputs["score"]),
                "score_props": _props(outputs["score"], _edges(outputs["score"])),
                "unc_mean": float(np.mean(outputs["unc"])),
                "unc_std": float(np.std(outputs["unc"]) or 1e-6),
                "class_dist": (np.bincount(outputs["pred"].astype(int)) / len(outputs["pred"])
                               if "proba" in outputs else None),
            }
        return prof


def feature_stats(profile: RefProfile, X: pd.DataFrame) -> pd.DataFrame:
    """Per-feature comparison of a production window against the reference profile."""
    rows = []
    for f in profile.feature_names:
        v = X[f].to_numpy(float)
        nan = np.isnan(v)
        c = v[~nan]
        ref = profile.ref_sorted[f]
        if len(c) < 5:
            rows.append(dict(feature=f, psi=5.0, ks=1.0, wd=5.0, js=1.0, miss=float(nan.mean()),
                             miss_delta=float(nan.mean() - profile.missing[f]), mean_shift=0.0,
                             log_scale=0.0, outlier=0.0))
            continue
        p_cur = _props(c, profile.edges[f])
        ks, wd = ks_wd_fast(ref, c)
        rows.append(dict(
            feature=f,
            psi=psi_from_props(profile.props[f], p_cur),
            ks=ks,
            wd=wd / profile.std[f],
            js=_js_props(profile.props[f], p_cur),
            miss=float(nan.mean()),
            miss_delta=float(nan.mean() - profile.missing[f]),
            mean_shift=float((c.mean() - profile.mean[f]) / profile.std[f]),
            log_scale=float(np.log((c.std() + 1e-9) / profile.std[f])),
            outlier=float(((c < profile.iqr_lo[f]) | (c > profile.iqr_hi[f])).mean()),
        ))
    return pd.DataFrame(rows).set_index("feature")


# ------------------------------------------------------------------------------------------------
def data_quality(profile: RefProfile, X: pd.DataFrame) -> dict:
    """Schema, missing-value, duplicate and outlier checks for one window."""
    missing_cols = sorted(set(profile.feature_names) - set(X.columns))
    extra_cols = sorted(set(X.columns) - set(profile.feature_names))
    fs = feature_stats(profile, X) if not missing_cols else None
    dup = float(X.duplicated().mean()) if len(X) else 0.0
    return {
        "schema_ok": not missing_cols,
        "missing_columns": missing_cols,
        "extra_columns": extra_cols,
        "missing_rate_mean": float(fs["miss"].mean()) if fs is not None else float("nan"),
        "missing_rate_max": float(fs["miss"].max()) if fs is not None else float("nan"),
        "duplicate_rate": dup,
        "outlier_rate": float(fs["outlier"].mean()) if fs is not None else float("nan"),
        "invalid_inf_rate": float(np.isinf(X.select_dtypes("number").to_numpy(float)).mean()),
    }
