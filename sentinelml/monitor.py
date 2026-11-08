"""Window-level monitoring signals and causal temporal feature engineering.

`window_signals` turns one production window into a flat dict of signals (drift, data quality,
model-output behaviour). `build_features` converts the *history* of those signals into predictive
features using only information available at time t (rolling windows over the past) - no leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .drift import RefProfile, feature_stats, _props, psi_from_props, js_divergence

# signal -> feature group (used for ablation studies)
SIGNAL_GROUPS = {
    "psi_mean": "drift", "psi_max": "drift", "ks_mean": "drift", "ks_max": "drift", "wd_mean": "drift",
    "wd_max": "drift", "js_mean": "drift", "imp_drift": "drift", "frac_drifted": "drift",
    "corr_shift": "drift", "scale_max": "drift", "meanshift_max": "drift",
    "missing_mean": "quality", "missing_max": "quality", "outlier_rate": "quality", "dup_rate": "quality",
    "unc_mean": "uncertainty", "unc_shift": "uncertainty", "pred_psi": "uncertainty",
    "pred_cls_js": "uncertainty",
    "loss_lag": "performance",
}
BASE_SIGNALS = list(SIGNAL_GROUPS)


def window_signals(profile: RefProfile, X: pd.DataFrame, out: dict) -> dict:
    fs = feature_stats(profile, X)
    imp = profile.importance / (profile.importance.sum() or 1.0)
    Xf = X[profile.feature_names].fillna(pd.Series(profile.median))
    c = np.nan_to_num(np.corrcoef(Xf.to_numpy().T))
    sig = {
        "psi_mean": fs.psi.mean(), "psi_max": fs.psi.max(),
        "ks_mean": fs.ks.mean(), "ks_max": fs.ks.max(),
        "wd_mean": fs.wd.mean(), "wd_max": fs.wd.max(), "js_mean": fs.js.mean(),
        "imp_drift": float((imp * fs.ks.to_numpy()).sum()),
        "frac_drifted": float((fs.psi > 0.2).mean()),
        "corr_shift": float(np.linalg.norm(c - profile.corr) / len(c)),
        "scale_max": float(fs.log_scale.abs().max()), "meanshift_max": float(fs.mean_shift.abs().max()),
        "missing_mean": fs.miss.mean(), "missing_max": fs.miss.max(),
        "outlier_rate": fs.outlier.mean(), "dup_rate": float(X.duplicated().mean()),
    }
    o = profile.out
    sig["unc_mean"] = float(np.mean(out["unc"]))
    sig["unc_shift"] = (sig["unc_mean"] - o["unc_mean"]) / o["unc_std"]
    sig["pred_psi"] = psi_from_props(o["score_props"], _props(out["score"], o["score_edges"]))
    if o.get("class_dist") is not None:
        cd = np.bincount(out["pred"].astype(int), minlength=len(o["class_dist"]))[: len(o["class_dist"])]
        p = np.clip(cd / max(cd.sum(), 1), 1e-4, None)
        q = np.clip(o["class_dist"], 1e-4, None)
        sig["pred_cls_js"] = float(0.5 * np.sum(p * np.log(2 * p / (p + q))) + 0.5 * np.sum(q * np.log(2 * q / (p + q))))
    else:
        sig["pred_cls_js"] = 0.0
    return {k: float(v) for k, v in sig.items()}


def build_features(sig: pd.DataFrame, temporal: bool = True, drop_groups: tuple = ()) -> pd.DataFrame:
    """Causal features for every time t from the signal history `sig` (index = window t).

    Per base signal s: s_t, 5-window mean, 10-window max, 1-step diff, 3-vs-3 trend, and a
    change-point z-score (mean of last 3 vs mean/std of the preceding 20 windows). Loss (performance)
    signals also get a rolling volatility. Everything looks backwards only.
    """
    cols = [c for c in BASE_SIGNALS if SIGNAL_GROUPS[c] not in drop_groups and c in sig]
    feats = {}
    for c in cols:
        s = sig[c]
        feats[c] = s
        if not temporal:
            continue
        m3 = s.rolling(3, min_periods=1).mean()
        prev = s.shift(3)
        feats[c + "_m5"] = s.rolling(5, min_periods=1).mean()
        feats[c + "_max10"] = s.rolling(10, min_periods=1).max()
        feats[c + "_d1"] = s.diff(1)
        feats[c + "_trend"] = m3 - m3.shift(3)
        feats[c + "_cpz"] = (m3 - prev.rolling(20, min_periods=5).mean()) / (prev.rolling(20, min_periods=5).std() + 1e-6)
        if c == "loss_lag":
            feats[c + "_vol10"] = s.rolling(10, min_periods=3).std()
    return pd.DataFrame(feats, index=sig.index)


def feature_group(name: str) -> str:
    base = name
    for suf in ("_m5", "_max10", "_d1", "_trend", "_cpz", "_vol10"):
        if name.endswith(suf):
            base = name[: -len(suf)]
            break
    return SIGNAL_GROUPS.get(base, "other")
