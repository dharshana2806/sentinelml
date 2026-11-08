"""Root-cause analysis engine.

Evidence layers (kept separate on purpose - they answer different questions):
  1. Statistical evidence  - which features changed distribution vs reference (KS, PSI, missing delta).
  2. Model attribution     - SHAP: did the *model's reliance* on a feature change (mean |SHAP| shift)?
  3. Counterfactual test   - "restoration": if feature j were restored to its reference distribution
                             (quantile mapping / median imputation), would labelled loss drop?
  4. Failure-type model    - a supervised classifier over diagnostic signatures (learned on the
                             failure-injection benchmark).
Only (3) is an *intervention on the model input*; none of them is causal identification of the real-world
cause. SHAP and drift statistics are associational. The output is a ranked list of hypotheses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .config import FAILURE_TYPES
from .drift import feature_stats, ks_statistic, _js_props


def quantile_restore(profile, x: np.ndarray, feature: str) -> np.ndarray:
    """Map window values of one feature onto the reference distribution (rank-preserving);
    NaNs are replaced by the reference median."""
    x = np.asarray(x, float).copy()
    nan = np.isnan(x)
    ref = profile.ref_sorted[feature]
    if (~nan).sum() >= 2:
        r = pd.Series(x[~nan]).rank(method="average").to_numpy()
        u = (r - 0.5) / len(r)
        grid = (np.arange(len(ref)) + 0.5) / len(ref)
        x[~nan] = np.interp(u, grid, ref)
    x[nan] = profile.median[feature]
    return x


def _mean_abs_shap(model, X: pd.DataFrame, max_rows: int = 150) -> np.ndarray:
    import shap
    Xi = model._prep(X.iloc[:max_rows])
    ex = shap.TreeExplainer(model.est)
    sv = ex.shap_values(Xi, check_additivity=False)
    if isinstance(sv, list):
        a = np.sum([np.abs(v).mean(0) for v in sv], axis=0)
    else:
        sv = np.asarray(sv)
        a = np.abs(sv).mean(0) if sv.ndim == 2 else np.abs(sv).mean(0).sum(-1)
    return a


def feature_diagnosis(ctx, X: pd.DataFrame, y: np.ndarray, use_shap: bool = True) -> pd.DataFrame:
    """Per-feature evidence table for a (labelled) production window set."""
    prof, model = ctx.profile, ctx.model
    fs = feature_stats(prof, X)
    base_loss = model.loss(y, model.predict(X))
    gains = {}
    for f in prof.feature_names:
        Xr = X.copy()
        Xr[f] = quantile_restore(prof, X[f].to_numpy(), f)
        gains[f] = base_loss - model.loss(y, model.predict(Xr))
    df = fs[["ks", "psi", "miss_delta", "outlier"]].copy()
    df["evidence"] = np.maximum(df.ks.to_numpy(), df.miss_delta.clip(lower=0).to_numpy())
    df["restore_gain"] = pd.Series(gains).clip(lower=0)
    df["importance"] = pd.Series(prof.importance, index=prof.feature_names)
    if use_shap:
        try:
            if not hasattr(ctx, "_shap_ref"):
                ctx._shap_ref = _mean_abs_shap(model, ctx.td.X_ref.sample(min(150, len(ctx.td.X_ref)), random_state=0))
            df["shap_shift"] = np.abs(_mean_abs_shap(model, X) - ctx._shap_ref)
        except Exception:                                    # shap optional / version issues
            df["shap_shift"] = np.nan
    df["imp_weighted"] = df.ks * df.importance
    ev = df.evidence / (df.evidence.max() or 1)
    rg = df.restore_gain / (df.restore_gain.max() or 1)
    df["combined"] = ev + rg
    return df


RANKERS = ["ks", "psi", "imp_weighted", "shap_shift", "restore_gain", "combined"]


def top_k(df: pd.DataFrame, ranker: str, k: int) -> list:
    s = df[ranker].fillna(-1)
    return list(s.sort_values(ascending=False, kind="stable").index[:k])


# ------------------------------------------------------------------------------------------------
SIG_COLS = ["psi_max", "psi_mean", "ks_max", "ks_mean", "wd_max", "missing_max", "outlier_rate", "scale_max",
            "meanshift_max", "unc_shift", "pred_psi", "pred_cls_js", "corr_shift", "frac_drifted"]


def signature(ctx, sig_rows: pd.DataFrame, X: pd.DataFrame, y: np.ndarray) -> dict:
    """Dataset-agnostic diagnostic signature of a degraded period (inputs of the failure-type model)."""
    s = {c: float(sig_rows[c].mean()) for c in SIG_COLS}
    s["psi_max"] = float(sig_rows["psi_max"].max())
    base, thr = ctx.base_loss, ctx.thr
    loss = ctx.model.loss(y, ctx.model.predict(X))
    s["loss_excess"] = (loss - base) / max(thr - base, 1e-6)
    if ctx.td.is_classification:
        k = max(ctx.n_classes, 2)
        p = np.clip(np.bincount(y.astype(int), minlength=k)[:k] / len(y), 1e-4, None)
        q = np.clip(np.bincount(ctx.td.y_train.astype(int), minlength=k)[:k] / len(ctx.td.y_train), 1e-4, None)
        s["label_js"] = _js_props(p, q)
    else:
        s["label_js"] = ks_statistic(ctx.td.y_train, y)
    s["loss_per_drift"] = s["loss_excess"] / (s["ks_max"] + 0.05)
    return s


class FailureTypeClassifier:
    """Supervised classifier: diagnostic signature -> failure type (7 classes)."""

    def __init__(self, seed: int = 0):
        self.model = RandomForestClassifier(300, min_samples_leaf=2, random_state=seed, n_jobs=1,
                                            class_weight="balanced")
        self.columns: list = []

    def fit(self, sigs: pd.DataFrame, types) -> "FailureTypeClassifier":
        self.columns = list(sigs.columns)
        self.model.fit(sigs.fillna(0), types)
        return self

    def predict_proba(self, sigs: pd.DataFrame) -> pd.DataFrame:
        p = self.model.predict_proba(sigs[self.columns].fillna(0))
        return pd.DataFrame(p, columns=self.model.classes_, index=sigs.index)


def diagnose(ctx, X: pd.DataFrame, y: np.ndarray, sig_rows: pd.DataFrame, type_clf: FailureTypeClassifier | None = None,
             k: int = 5) -> dict:
    """Full diagnosis: ranked suspect features + failure-type probabilities."""
    df = feature_diagnosis(ctx, X, y)
    ranked = df.sort_values("combined", ascending=False).head(k)
    out = {"suspect_features": [
        {"feature": f, **{c: (None if pd.isna(r[c]) else round(float(r[c]), 4))
                           for c in ("combined", "ks", "miss_delta", "restore_gain", "shap_shift", "importance")}}
        for f, r in ranked.iterrows()]}
    sig = signature(ctx, sig_rows, X, y)
    out["signature"] = {k_: round(v, 4) for k_, v in sig.items()}
    if type_clf is not None:
        p = type_clf.predict_proba(pd.DataFrame([sig])).iloc[0].sort_values(ascending=False)
        out["failure_type_probabilities"] = {k_: round(float(v), 4) for k_, v in p.items()}
    out["caveat"] = ("Ranked hypotheses from statistical, attribution and counterfactual-restoration evidence; "
                     "not causal identification.")
    return out
